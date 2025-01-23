// Copyright 2025 The Soda Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"embed"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	"github.com/pointlander/soda/vector"
)

const (
	// Size is the number of histograms
	Size = 8
	// Order is the order of the markov model
	Order = 7
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Matrix is a float64 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new float32 matrix
func NewMatrix(cols, rows int, data ...float64) Matrix {
	if data == nil {
		data = make([]float64, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix) MulT(n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float32 matrices
func (m Matrix) Add(n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix) Softmax(T float64) Matrix {
	output := NewMatrix(m.Cols, m.Rows)
	max := 0.0
	for _, v := range m.Data {
		v /= T
		if v > max {
			max = v
		}
	}
	s := max * S
	for i := 0; i < len(m.Data); i += m.Cols {
		sum := 0.0
		values := make([]float64, m.Cols)
		for j, value := range m.Data[i : i+m.Cols] {
			values[j] = math.Exp(value/T - s)
			sum += values[j]
		}
		for _, value := range values {
			output.Data = append(output.Data, value/sum)
		}
	}
	return output
}

// Entropy calculates the entropy of the matrix rows
func (m Matrix) Entropy() Matrix {
	output := NewMatrix(m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := 0.0
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * math.Log(value)
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}

// T tramsposes a matrix
func (m Matrix) T() Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

func dot(x, y []float64) (z float64) {
	for i := range x {
		z += x[i] * y[i]
	}
	return z
}

func softmax(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(input Matrix, output *[256]float32) {
	values := make([]float64, input.Rows)
	V := input.T()
	sums := make([]float64, V.Rows)
	for i := 0; i < input.Rows; i++ {
		K := input.Data[i*input.Cols : (i+1)*input.Cols]
		for j := 0; j < input.Rows; j++ {
			Q := input.Data[j*input.Cols : (j+1)*input.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			sums[j] += dot(values, V)
		}
	}
	for i, v := range sums {
		output[i] = float32(v)
	}
}

// Markov is a markov model
type Markov [Order + 1]byte

// Histogram is a buffered histogram
type Histogram struct {
	Vector [256]byte
	Buffer [128]byte
	Index  int
	Size   int
}

// NewHistogram make a new histogram
func NewHistogram(size int) Histogram {
	h := Histogram{
		Size: size,
	}
	return h
}

// Add adds a symbol to the histogram
func (h *Histogram) Add(s byte) {
	index := (h.Index + 1) % h.Size
	if symbol := h.Buffer[index]; h.Vector[symbol] > 0 {
		h.Vector[symbol]--
	}
	h.Buffer[index] = s
	h.Vector[s]++
	h.Index = index
}

// Mixer mixes several histograms together
type Mixer struct {
	Markov     Markov
	Histograms []Histogram
}

// NewMixer makes a new mixer
func NewMixer() Mixer {
	histograms := make([]Histogram, Size)
	histograms[0] = NewHistogram(1)
	histograms[1] = NewHistogram(2)
	histograms[2] = NewHistogram(4)
	histograms[3] = NewHistogram(8)
	histograms[4] = NewHistogram(16)
	histograms[5] = NewHistogram(32)
	histograms[6] = NewHistogram(64)
	histograms[7] = NewHistogram(128)
	return Mixer{
		Histograms: histograms,
	}
}

func (m Mixer) Copy() Mixer {
	histograms := make([]Histogram, Size)
	for i := range m.Histograms {
		histograms[i] = m.Histograms[i]
	}
	return Mixer{
		Markov:     m.Markov,
		Histograms: histograms,
	}
}

// Add adds a symbol to a mixer
func (m *Mixer) Add(s byte) {
	for i := range m.Histograms {
		m.Histograms[i].Add(s)
	}
	for k := Order; k > 0; k-- {
		m.Markov[k] = m.Markov[k-1]
	}
	m.Markov[0] = s
}

// Mix mixes the histograms outputting a matrix
func (m Mixer) Mix(output *[256]float32) {
	x := NewMatrix(256, Size)
	for i := range m.Histograms {
		sum := 0.0
		for _, v := range m.Histograms[i].Vector {
			sum += float64(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float64(v)/sum)
		}
	}
	SelfAttention(x, output)
}

//go:embed books/*
var Data embed.FS

var (
	// FlagQuery is the query string
	FlagQuery = flag.String("query", "What is the meaning of life?", "query flag")
	// FlagBuild build the database
	FlagBuild = flag.Bool("build", false, "build the database")
	// FlagServer is server mode
	FlagServer = flag.Bool("server", false, "server mode")
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

func sqrt(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

// CS is float32 cosine similarity
func CS(a []float32, b []float32) float32 {
	aa, bb, ab := vector.Dot(a, a), vector.Dot(b, b), vector.Dot(a, b)
	return float32(ab / (sqrt(aa) * sqrt(bb)))
}

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Scale is the scale of the model
	Scale = 128
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Handler is a http handler
type Handler struct {
	Header Header
	Sizes  []uint64
	Sums   []uint64
}

func (h Handler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	query, err := io.ReadAll(request.Body)
	if err != nil {
		panic(err)
	}
	request.Body.Close()
	output := h.Header.Soda(h.Sizes, h.Sums, query)
	str := query
	for i := range output {
		str = append(str, output[i].Symbol)
	}
	response.Write(str)
}

func main() {
	flag.Parse()

	if *FlagBuild {
		Build()
		return
	} else if *FlagServer {
		header, sizes, sums := LoadHeader()
		infer := Handler{
			Header: header,
			Sizes:  sizes,
			Sums:   sums,
		}
		assetsHandler := http.FileServer(http.Dir("assets"))
		mux := http.NewServeMux()
		mux.Handle("/infer", infer)
		mux.Handle("/", assetsHandler)
		s := &http.Server{
			Addr:           ":8080",
			Handler:        mux,
			ReadTimeout:    10 * time.Second,
			WriteTimeout:   10 * time.Second,
			MaxHeaderBytes: 1 << 20,
		}
		err := s.ListenAndServe()
		if err != nil {
			fmt.Println("Failed to start server", err)
			return
		}
		return
	}

	header, sizes, sums := LoadHeader()
	output := header.Soda(sizes, sums, []byte(*FlagQuery))
	str := []byte(*FlagQuery)
	for i := range output {
		str = append(str, output[i].Symbol)
	}
	fmt.Println(string(str))
}
