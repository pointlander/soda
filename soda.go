// Copyright 2025 The Soda Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/pointlander/gradient/tf32"

	"github.com/alixaxel/pagerank"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// ModelSize is the model size
	ModelSize = 8
	// HeaderLineSize is the size of a header line
	HeaderLineSize = 4*256 + 1*8
	// EntryLineSize is the size of an entry line
	EntryLineSize = 4*256 + 1 + 8
	// Offset is the offset to the entries
	Offset = ModelSize * 1024 * HeaderLineSize
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Vector is a vector
type Vector struct {
	Vector [256]float32
	Symbol uint64
	Next   uint64
}

// Bucket is a bucket of vectors
type Bucket struct {
	Vector  [256]float32
	Vectors uint64
	Count   int
}

// Output is the output of the model
type Output struct {
	Index  uint64 `json:"index"`
	Symbol uint8  `json:"-"`
	S      string `json:"symbol"`
}

// Result is an index search result
type Result struct {
	Index  int
	Vector uint64
}

func process(done chan Result, model []Bucket, pool []Vector, vector uint64) {
	query, index, max := pool[vector].Vector[:], 0, float32(0.0)
	for i := range model {
		cs := CS(query, model[i].Vector[:])
		if cs > max {
			max, index = cs, i
		}
	}
	done <- Result{
		Index:  index,
		Vector: vector,
	}
}

// Header is an index
type Header []Bucket

// NewHeader generates a new header
func NewHeader(data []byte) Header {
	model := make(Header, ModelSize*1024)
	rng := rand.New(rand.NewSource(1))

	avg := make([]float64, 256)
	m := NewMixer()
	m.Add(0)
	for _, v := range data {
		var vector [256]float32
		m.Mix(&vector)
		for i, v := range vector {
			avg[i] += float64(v)
		}
		m.Add(v)
	}
	for i := range avg {
		avg[i] /= float64(len(data))
	}
	cov := [256][256]float64{}
	m = NewMixer()
	m.Add(0)
	for _, v := range data {
		var vector [256]float32
		m.Mix(&vector)
		for i, v := range vector {
			for ii, vv := range vector {
				diff1 := avg[i] - float64(v)
				diff2 := avg[ii] - float64(vv)
				cov[i][ii] += diff1 * diff2
			}
		}
		m.Add(v)
	}
	for i := range cov {
		for j := range cov[i] {
			cov[i][j] = cov[i][j] / float64(len(data))
		}
	}
	fmt.Println(avg)

	set := tf32.NewSet()
	set.Add("A", 256, 256)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	others := tf32.NewSet()
	others.Add("E", 256, 256)
	E := others.ByName["E"]
	for i := range cov {
		for j := range cov[i] {
			E.X = append(E.X, float32(cov[i][j]))
		}
	}

	loss := tf32.Sum(tf32.Quadratic(others.Get("E"), tf32.Mul(set.Get("A"), set.Get("A"))))

	points := make(plotter.XYs, 0, 8)
	for i := 0; i < 1024; i++ {
		pow := func(x float32) float32 {
			y := math.Pow(float64(x), float64(i+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return float32(y)
		}

		set.Zero()
		others.Zero()
		cost := tf32.Gradient(loss).X[0]
		if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
			fmt.Println(i, cost)
			break
		}

		norm := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = float32(math.Sqrt(float64(norm)))
		b1, b2 := pow(B1), pow(B2)
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range set.Weights {
			for l, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][l] + (1-B1)*g
				v := B2*w.States[StateV][l] + (1-B2)*g*g
				w.States[StateM][l] = m
				w.States[StateV][l] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}
		points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
		fmt.Println(i, cost)
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "epochs.png")
	if err != nil {
		panic(err)
	}

	A := NewMatrix(256, 256)
	for _, v := range set.ByName["A"].X {
		A.Data = append(A.Data, float64(v))
	}
	u := NewMatrix(256, 1, avg...)
	fmt.Println(ModelSize * 1024 * 512 * 4.0 / (1024.0 * 1024.0 * 1024.0))
	for i := range model {
		z := NewMatrix(256, 1)
		for j := 0; j < 256; j++ {
			z.Data = append(z.Data, rng.NormFloat64())
		}
		x := A.MulT(z).Add(u)
		for j, v := range x.Data {
			model[i].Vector[j] = float32(v)
		}
	}
	return model
}

// LoadHeader loads the header
func LoadHeader() (Header, []uint64, []uint64) {
	model := make(Header, ModelSize*1024)
	sizes := make([]uint64, ModelSize*1024)
	in, err := os.Open("db.bin")
	if err != nil {
		panic(err)
	}
	defer in.Close()

	buffer32 := make([]byte, 4)
	buffer64 := make([]byte, 8)
	for i := range model {
		for j := range model[i].Vector {
			n, err := in.Read(buffer32)
			if err != nil {
				panic(err)
			}
			if n != len(buffer32) {
				panic("4 bytes should have been read")
			}
			var bits uint32
			for i := range buffer32 {
				bits |= uint32(buffer32[i]) << (8 * i)
			}
			model[i].Vector[j] = math.Float32frombits(bits)
		}
		var count uint64
		n, err := in.Read(buffer64)
		if err != nil {
			panic(err)
		}
		if n != len(buffer64) {
			panic("4 bytes should have been read")
		}
		for i := range buffer64 {
			count |= uint64(buffer64[i]) << (8 * i)
		}
		sizes[i] = count
	}
	sums, sum := make([]uint64, len(sizes)), uint64(0)
	for i, v := range sizes {
		sums[i] = sum
		sum += v
	}
	return model, sizes, sums
}

// Build builds the model
func Build() {
	cpus := runtime.NumCPU()
	file, err := Data.Open("books/10.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	input, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}
	data := input
	counts := make([]uint64, len(data))
	{
		str := string(data)
		runes := []rune(str)
		index := 0
		for j, r := range runes {
			size := utf8.RuneLen(r)
			for i := 0; i < size; i++ {
				counts[index] = uint64(j)
				index++
			}
		}
	}

	model := NewHeader(data)
	pool, item := make([]Vector, len(data)+1), uint64(1)

	done, m, index, flight := make(chan Result, cpus), NewMixer(), 0, 0
	m.Add(0)
	for index < len(data) && flight < cpus {
		symbol := data[index]
		m.Mix(&pool[item].Vector)
		pool[item].Symbol = uint64(index)
		go process(done, model, pool, item)
		item++
		m.Add(symbol)
		flight++
		index++
	}
	for index < len(data) {
		result := <-done
		flight--
		pool[result.Vector].Next = model[result.Index].Vectors
		model[result.Index].Vectors = result.Vector
		model[result.Index].Count++

		symbol := data[index]
		m.Mix(&pool[item].Vector)
		pool[item].Symbol = uint64(index)
		go process(done, model, pool, item)
		item++
		m.Add(symbol)
		flight++
		index++
		if index%8 == 0 {
			fmt.Println(index, "/", len(data), "=", float64(index)/float64(len(data)))
		}
		if index%128 == 0 {
			runtime.GC()
		}
	}
	for i := 0; i < flight; i++ {
		result := <-done
		pool[result.Vector].Next = model[result.Index].Vectors
		model[result.Index].Vectors = result.Vector
		model[result.Index].Count++
	}

	db, err := os.Create("db.bin")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	buffer32 := make([]byte, 4)
	buffer64 := make([]byte, 8)
	for i := range model {
		for _, v := range model[i].Vector {
			bits := math.Float32bits(v)
			for i := range buffer32 {
				buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := db.Write(buffer32)
			if err != nil {
				panic(err)
			}
			if n != len(buffer32) {
				panic("4 bytes should be been written")
			}
		}
		count := uint64(model[i].Count)
		for i := range buffer64 {
			buffer64[i] = byte((count >> (8 * i)) & 0xFF)
		}
		n, err := db.Write(buffer64)
		if err != nil {
			panic(err)
		}
		if n != len(buffer64) {
			panic("8 bytes should be been written")
		}
	}

	symbol := make([]byte, 1)
	for i := range model {
		vector := model[i].Vectors
		for vector != 0 {
			for _, v := range pool[vector].Vector {
				bits := math.Float32bits(v)
				for i := range buffer32 {
					buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
				}
				n, err := db.Write(buffer32)
				if err != nil {
					panic(err)
				}
				if n != len(buffer32) {
					panic("4 bytes should be been written")
				}
			}
			symbol[0] = data[pool[vector].Symbol]
			n, err := db.Write(symbol)
			if err != nil {
				panic(err)
			}
			if n != len(symbol) {
				panic("1 bytes should be been written")
			}

			for i := range buffer64 {
				buffer64[i] = byte((counts[pool[vector].Symbol] >> (8 * i)) & 0xFF)
			}
			n, err = db.Write(buffer64)
			if err != nil {
				panic(err)
			}
			if n != len(buffer64) {
				panic("8 bytes should be been written")
			}
			vector = pool[vector].Next
		}
	}
}

// Soda is the soda model
func (h Header) Soda(sizes, sums []uint64, query []byte) (output []Output) {
	cpus := runtime.NumCPU()
	rng := rand.New(rand.NewSource(1))
	in := make([]*os.File, cpus)
	for i := range in {
		var err error
		in[i], err = os.Open("db.bin")
		if err != nil {
			panic(err)
		}
	}
	defer func() {
		for i := range in {
			in[i].Close()
		}
	}()

	vectors := []*[256]float32{}
	m := NewMixer()
	for _, v := range query {
		m.Add(v)
		var vector [256]float32
		vec := &vector
		vectors = append(vectors, vec)
		m.Mix(vec)
		cp := make([]float64, len(vector))
		for i := range cp {
			cp[i] = float64(vector[i])
		}
	}

	type Result struct {
		Output
		CS     float32
		Vector []float32
	}
	done := make(chan []Result, 8)
	search := func(r, index int, data []float32) {
		buffer := make([]byte, sizes[index]*EntryLineSize)
		_, err := in[r].Seek(int64(Offset+sums[index]*EntryLineSize), io.SeekStart)
		if err != nil {
			panic(err)
		}
		n, err := in[r].Read(buffer)
		if err != nil {
			panic(err)
		}
		if n != len(buffer) {
			panic(fmt.Sprintf("%d bytes should have been read", len(buffer)))
		}
		candidates := make([]Result, sizes[index])
		for j := 0; j < int(sizes[index]); j++ {
			vector := make([]float32, 256)
			for k := range vector {
				var bits uint32
				for l := 0; l < 4; l++ {
					bits |= uint32(buffer[j*EntryLineSize+4*k+l]) << (8 * l)
				}
				vector[k] = math.Float32frombits(bits)
			}
			cs := CS(vector, data)
			max, symbolIndex, symbol := cs, uint64(0), buffer[(j+1)*EntryLineSize-1-8]
			for k := 0; k < 8; k++ {
				symbolIndex |= uint64(buffer[(j+1)*EntryLineSize-8+k]) << (8 * k)
			}
			candidates[j] = Result{
				Output: Output{
					Index:  symbolIndex,
					Symbol: symbol,
				},
				CS:     max,
				Vector: vector,
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].CS > candidates[j].CS
		})
		size := uint64(64)
		if sizes[index] < size {
			size = sizes[index]
		}
		results := make([]Result, size)
		copy(results, candidates[:size])
		done <- results
	}

	result := make([]Output, 0, 8)
	var symbols []byte
	for i := 0; i < *FlagCount; i++ {
		var data [256]float32
		vec := &data
		vectors = append(vectors, vec)
		m.Mix(vec)
		type Index struct {
			Index int
			Value float32
		}
		indexes := make([]Index, len(h))
		for i := range h {
			if sizes[i] == 0 {
				continue
			}
			indexes[i].Index = i
			indexes[i].Value = CS(h[i].Vector[:], data[:])
		}
		sort.Slice(indexes, func(i, j int) bool {
			return indexes[i].Value > indexes[j].Value
		})

		var results []Result
		for j := 0; j < cpus; j++ {
			go search(j, indexes[j].Index, data[:])
		}
		for j := 0; j < cpus; j++ {
			result := <-done
			results = append(results, result...)
		}
		sort.Slice(results, func(i, j int) bool {
			return results[i].CS > results[j].CS
		})

		length := len(vectors) + len(results)
		graph := pagerank.NewGraph()
		for j := 0; j < length; j++ {
			for k := 0; k < length; k++ {
				var x, y []float32
				if j < len(vectors) {
					x = (*vectors[j])[:]
				} else {
					x = results[j-len(vectors)].Vector
				}
				if k < len(vectors) {
					y = (*vectors[k])[:]
				} else {
					y = results[k-len(vectors)].Vector
				}
				cs := CS(x, y)
				graph.Link(uint32(i), uint32(j), float64(cs))
			}
		}
		ranks := make([]float64, length)
		graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
			ranks[node] = rank
		})
		index, total := 0, 0.0
		for j := len(vectors); j < length; j++ {
			total += ranks[j]
		}
		sum, selection := 0.0, rng.Float64()
		for j := len(vectors); j < length; j++ {
			if selection < sum {
				index = j
				break
			}
			sum += ranks[j] / sum
		}
		index -= len(vectors)

		m.Add(results[index].Symbol)
		symbols = append(symbols, results[index].Symbol)
		if utf8.FullRune(symbols) {
			results[index].S = string(symbols)
			symbols = []byte{}
			result = append(result, results[index].Output)
		}
	}

	return result
}
