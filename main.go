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
	"net/http"
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
	// FlagServer is server mode
	FlagServer = flag.Bool("server", false, "server mode")
	// FlagBrute is the brute force mode
	FlagBrute = flag.Bool("brute", false, "brute force mode")
)

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
		mux := http.NewServeMux()
		mux.Handle("/infer", infer)
		mux.Handle("/bible", Bible{})
		mux.Handle("/index.html", Root{})
		mux.Handle("/", Root{})
		s := &http.Server{
			Addr:           ":8080",
			Handler:        mux,
			ReadTimeout:    30 * time.Second,
			WriteTimeout:   30 * time.Second,
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
