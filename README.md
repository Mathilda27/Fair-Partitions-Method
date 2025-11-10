# Fair Partitions Method

From a fixed convex polygon and n random points in R², we construct a Voronoi partition of the original polygon and an associated Centroidal Voronoi partition via Lloyd´s Algorithm. We then proceed to apply Normal Flow Algorithm to minimize the error in areas and perimeters. This results in a partition of the original polygon into convex regions having equal area and equal perimeter.

## Table of Contents
- [Fair Partitions Method](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [License](#license)

## Introduction

Problems regarding partitions of regions are common optimization problems as they arise from real world applications. Many authors have addressed the problem of partitioning convex regions with specific properties. Among these, one particular that stands out is partitioning a convex polygon into convex regions of equal area and equal perimeter.
This problem was firstly proposed by Nandakumar and RamanaRao where such partition was coined Convex Fair Partition. In this project we provide a code that solves the Fair Partition problem without imposing any constraints. For further information please visit https://doi.org/10.21203/rs.3.rs-3276690/v1 and/or https://doi.org/10.1016/j.mex.2023.102530

## Features

- Partition a convex polygon into regions of equal area and equal perimeter.

## Getting Started

Download the class_Equipartition.py file

### Prerequisites

Import libraries from Dependencies.py File.

## Usage

Copy the example in the Example.py File, paste it in the class_Equipartition.py File and run it in a python terminal.

## License

This project is under MIT License. If you wish to use or reproduce this code please give credit to Bernardo Uribe and Mathilda Campillo.





