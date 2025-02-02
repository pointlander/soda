// Copyright 2025 The Soda Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"time"
)

//go:embed books/*
var Data embed.FS

//go:embed assets/index.html
var Index embed.FS

var (
	// FlagQuery is the query string
	FlagQuery = flag.String("query", "What is the meaning of life?", "query flag")
	// FlagCount count is the number of symbols to generate
	FlagCount = flag.Int("count", 128, "number of symbols to generate")
	// FlagBuild build the database
	FlagBuild = flag.Bool("build", false, "build the database")
	// FlagMoar use more training data
	FlagMoar = flag.Bool("moar", false, "use more training data")
	// FlagServer is server mode
	FlagServer = flag.Bool("server", false, "server mode")
	// FlagBrute is the brute force mode
	FlagBrute = flag.Bool("brute", false, "brute force mode")
	// FlagRank is page rank mode
	FlagRank = flag.Bool("rank", false, "page rank mode")
)

var Moar = []string{
	"books/84.txt.utf-8.bz2",    // 2 Frankenstein; Or, The Modern Prometheus
	"books/2701.txt.utf-8.bz2",  // 3 Moby Dick; Or, The Whale
	"books/1513.txt.utf-8.bz2",  // 4 Romeo and Juliet
	"books/1342.txt.utf-8.bz2",  // 5 Pride and Prejudice
	"books/11.txt.utf-8.bz2",    // 6 Alice's Adventures in Wonderland
	"books/145.txt.utf-8.bz2",   // 7 Middlemarch
	"books/2641.txt.utf-8.bz2",  // 8 A Room with a View
	"books/37106.txt.utf-8.bz2", // 9 Little Women; Or, Meg, Jo, Beth, and Amy
	"books/64317.txt.utf-8.bz2", // 10 The Great Gatsby
	"books/100.txt.utf-8.bz2",   // 11 The Complete Works of William Shakespeare
	"books/75256.txt.utf-8.bz2", // 12 Pirate tales from the law
	"books/16389.txt.utf-8.bz2", // 13 The Enchanted April
	"books/67979.txt.utf-8.bz2", // 14 The Blue Castle: a novel
	"books/394.txt.utf-8.bz2",   // 15 Cranford
	"books/6761.txt.utf-8.bz2",  // 16 The Adventures of Ferdinand Count Fathom — Complete
	"books/2542.txt.utf-8.bz2",  // 17 A Doll's House : a play
	"books/2160.txt.utf-8.bz2",  // 18 The Expedition of Humphry Clinker
	"books/4085.txt.utf-8.bz2",  // 19 The Adventures of Roderick Random
	"books/6593.txt.utf-8.bz2",  // 20 History of Tom Jones, a Foundling
}

// Root is the root file
type Root struct{}

// ServeHTTP implements model inference access
func (r Root) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	file, err := Index.Open("assets/index.html")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	input, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}
	response.Header().Set("Content-Type", "text/html; charset=utf-8")
	response.Write(input)
}

// Bibiel is the bible file
type Bible struct{}

// ServeHTTP implements model inference access
func (b Bible) ServeHTTP(response http.ResponseWriter, request *http.Request) {
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
	if *FlagMoar {
		for _, f := range Moar {
			file, err := Data.Open(f)
			if err != nil {
				panic(err)
			}
			defer file.Close()
			reader := bzip2.NewReader(file)
			data, err := io.ReadAll(reader)
			if err != nil {
				panic(err)
			}
			input = append(input, data...)
		}
	}
	response.Header().Set("Content-Type", "text/plain; charset=utf-8")
	response.Write(input)
}

// Handler is a http handler
type Handler struct {
	Header Header
	Sizes  []uint64
	Sums   []uint64
}

// ServeHTTP implements model inference access
func (h Handler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	query, err := io.ReadAll(request.Body)
	if err != nil {
		panic(err)
	}
	request.Body.Close()
	searches := h.Header.Soda(h.Sizes, h.Sums, query)
	data, err := json.Marshal(searches[0].Result)
	if err != nil {
		panic(err)
	}
	response.Header().Set("Content-Type", "application/json; charset=utf-8")
	response.Write(data)
}

// Brute is brute force mode
func Brute() {
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

	type Vector struct {
		Vector [Size]float32
		Symbol byte
	}
	vectors := make([]Vector, len(input))
	m := NewMixer()
	m.Add(0)
	vector := make([]float32, Size)
	for i, v := range input {
		m.MixEntropy(vector)
		copy(vectors[i].Vector[:], vector)
		vectors[i].Symbol = v
		m.Add(v)
	}

	query := []byte("Go")
	m = NewMixer()
	for _, v := range query {
		m.Add(v)
	}

	m.MixEntropy(vector)
	index, max := 0, float32(0.0)
	for i := range vectors {
		cs := CS(vector, vectors[i].Vector[:])
		if cs > max {
			max, index = cs, i
			fmt.Printf("%d %f %d %c\n", index, max, vectors[index].Symbol, vectors[index].Symbol)
		}
	}
}

// Rank is page rank mode
func Rank() {
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

	type Entry struct {
		Vector [Size]float32
		Symbol byte
		Index  uint64
	}

	if *FlagBuild {
		model := make([]Entry, len(input))
		m := NewMixer()
		m.Add(0)
		for i, v := range input {
			m.MixRank(&model[i].Vector)
			model[i].Symbol = v
			model[i].Index = uint64(i)
			m.Add(v)
			fmt.Println(i, "/", len(input))
		}

		db, err := os.Create("rdb.bin")
		if err != nil {
			panic(err)
		}
		defer db.Close()

		buffer32 := make([]byte, 4)
		buffer64 := make([]byte, 8)
		symbol := make([]byte, 1)
		for i := range model {
			vector := model[i].Vector
			for _, v := range vector {
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
			symbol[0] = model[i].Symbol
			n, err := db.Write(symbol)
			if err != nil {
				panic(err)
			}
			if n != len(symbol) {
				panic("1 bytes should be been written")
			}

			for i := range buffer64 {
				buffer64[i] = byte((model[i].Index >> (8 * i)) & 0xFF)
			}
			n, err = db.Write(buffer64)
			if err != nil {
				panic(err)
			}
			if n != len(buffer64) {
				panic("8 bytes should be been written")
			}
		}

		return
	}

	m := NewMixer()
	for _, v := range []byte(*FlagQuery) {
		m.Add(v)
	}

	db, err := os.Open("rdb.bin")
	if err != nil {
		panic(err)
	}
	defer db.Close()

	buffer, err := io.ReadAll(db)
	if err != nil {
		panic(err)
	}

	const EntryLineSize = 8*4 + 1 + 8
	model := make([]Entry, len(input))
	for j := range model {
		vector := [Size]float32{}
		for k := range vector {
			var bits uint32
			for l := 0; l < 4; l++ {
				bits |= uint32(buffer[j*EntryLineSize+4*k+l]) << (8 * l)
			}
			vector[k] = math.Float32frombits(bits)
		}
		symbolIndex, symbol := uint64(0), buffer[(j+1)*EntryLineSize-1-8]
		for k := 0; k < 8; k++ {
			symbolIndex |= uint64(buffer[(j+1)*EntryLineSize-8+k]) << (8 * k)
		}
		model[j].Vector = vector
		model[j].Symbol = symbol
		model[j].Index = symbolIndex
	}

	symbols := []byte{}
	for i := 0; i < 128; i++ {
		max, vector, symbol := float32(0.0), [Size]float32{}, byte(0)
		m.MixRank(&vector)
		for j := range model {
			cs := CS(vector[:], model[j].Vector[:])
			if cs > max {
				max, symbol = cs, model[j].Symbol
			}
		}
		symbols = append(symbols, symbol)
		m.Add(symbol)
	}
	fmt.Println(string(symbols))
}

func main() {
	flag.Parse()

	if *FlagRank {
		Rank()
		return
	} else if *FlagBuild {
		Build()
		return
	} else if *FlagServer {
		header, sizes, sums := LoadHeader()
		infer := Handler{
			Header: header,
			Sizes:  sizes,
			Sums:   sums,
		}
		mux := http.NewServeMux()
		mux.Handle("/infer", infer)
		mux.Handle("/bible", Bible{})
		mux.Handle("/index.html", Root{})
		mux.Handle("/", Root{})
		s := &http.Server{
			Addr:           ":8080",
			Handler:        mux,
			ReadTimeout:    10 * 60 * time.Second,
			WriteTimeout:   10 * 60 * time.Second,
			MaxHeaderBytes: 1 << 20,
		}
		err := s.ListenAndServe()
		if err != nil {
			fmt.Println("Failed to start server", err)
			return
		}
		return
	} else if *FlagBrute {
		Brute()
		return
	}

	header, sizes, sums := LoadHeader()
	searches := header.Soda(sizes, sums, []byte(*FlagQuery))
	for _, search := range searches {
		output := search.Result
		str := []byte(*FlagQuery)
		for i := range output {
			str = append(str, output[i].Symbol)
		}
		fmt.Println(string(str))
		fmt.Println(search.Rank, " ---------------------------------------")
	}
}
