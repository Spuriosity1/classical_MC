#!/usr/bin/env python3

import re
import sys
import os
import glob
import argparse
from collections import defaultdict

import numpy as np
import h5py

# Usage: acc_runs_2.py <file1> <file2> ...
# Parses the metadata for each file, then plots the three symmetry-equivalnt Bragg peaks as a funciton of a configurable 'x' parameter.
# 
