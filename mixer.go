// Copyright 2025 The Soda Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/alixaxel/pagerank"
	"github.com/pointlander/soda/vector"
)

const (
	// Size is the number of histograms
	Size = 8
	// Order is the order of the markov model
	Order = 7
)

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
		sum := float32(0.0)
		for _, v := range m.Histograms[i].Vector {
			sum += float32(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float32(v)/sum)
		}
	}
	SelfAttention(x, output)
}

// MixEntropy mixes the histograms and outputs entropy
func (m Mixer) MixEntropy(output []float32) {
	x := NewMatrix(256, Size)
	for i := range m.Histograms {
		sum := float32(0.0)
		for _, v := range m.Histograms[i].Vector {
			sum += float32(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float32(v)/sum)
		}
	}
	SelfEntropy(x, output)
	aa := sqrt(vector.Dot(output, output))
	for i, v := range output {
		output[i] = v / aa
	}
}

// MixRank mixes the histograms and outputs page rank
func (m Mixer) MixRank(output *[Size]float32) {
	x := NewMatrix(256, Size)
	for i := range m.Histograms {
		sum := float32(0.0)
		for _, v := range m.Histograms[i].Vector {
			sum += float32(v)
		}
		for _, v := range m.Histograms[i].Vector {
			x.Data = append(x.Data, float32(v)/sum)
		}
	}
	graph := pagerank.NewGraph()
	for i := 0; i < Size; i++ {
		a := x.Data[i*256 : i*256+256]
		for j := 0; j < Size; j++ {
			b := x.Data[j*256 : j*256+256]
			cs := CS(a, b)
			graph.Link(uint32(i), uint32(j), float64(cs))
		}
	}
	graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
		output[node] = float32(rank)
	})
	a := output[:]
	aa := sqrt(vector.Dot(a, a))
	for i, v := range output {
		output[i] = v / aa
	}
}
