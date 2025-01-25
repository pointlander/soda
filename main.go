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
	// FlagBuild build the database
	FlagBuild = flag.Bool("build", false, "build the database")
	// FlagServer is server mode
	FlagServer = flag.Bool("server", false, "server mode")
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
	output := h.Header.Soda(h.Sizes, h.Sums, query)
	data, err := json.Marshal(output)
	if err != nil {
		panic(err)
	}
	response.Write(data)
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
