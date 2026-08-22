from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SingletonTZCYX:
    """Read C/Y/X tiles while lazily selecting singleton OME T/Z axes.

    Integer indexing removes T, Z and C before NumPy conversion, so a Zarr
    backend decompresses only chunks intersecting the requested Y/X region.
    Non-singleton T/Z axes are rejected because silently choosing a timepoint or
    focal plane would change the scientific meaning of the analysis.
    """

    def __init__(self, array, axes: Sequence[str]):
        self.array = array
        self.axes = [str(axis).upper() for axis in axes]
        self.shape = tuple(int(value) for value in array.shape)
        if len(self.axes) != len(self.shape):
            raise RuntimeError(
                f'OME-Zarr axes {self.axes} do not match array shape {self.shape}.'
            )
        for required in ('C', 'Y', 'X'):
            if self.axes.count(required) != 1:
                raise RuntimeError(
                    f'OME-Zarr must contain exactly one {required} axis; got {self.axes}.'
                )
        unsupported = [axis for axis in self.axes if axis not in {'T', 'Z', 'C', 'Y', 'X'}]
        if unsupported:
            raise RuntimeError(f'Unsupported OME-Zarr axes {unsupported}; full axes={self.axes}.')
        for singleton in ('T', 'Z'):
            if singleton in self.axes:
                size = self.shape[self.axes.index(singleton)]
                if size != 1:
                    raise RuntimeError(
                        f'OME-Zarr {singleton} axis has size {size}; only singleton T/Z axes '
                        'can be analysed without an explicit plane-selection policy.'
                    )

    @property
    def shape_cyx(self) -> tuple[int, int, int]:
        return tuple(self.shape[self.axes.index(axis)] for axis in ('C', 'Y', 'X'))

    def read(self, channel: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        channels, _, _ = self.shape_cyx
        channel = int(channel)
        if channel < 0 or channel >= channels:
            raise RuntimeError(f'Channel {channel} is invalid for {channels} channels.')
        selector = []
        remaining_axes = []
        for axis in self.axes:
            if axis in {'T', 'Z'}:
                selector.append(0)
            elif axis == 'C':
                selector.append(channel)
            elif axis == 'Y':
                selector.append(y_slice); remaining_axes.append('Y')
            elif axis == 'X':
                selector.append(x_slice); remaining_axes.append('X')
        tile = np.asarray(self.array[tuple(selector)])
        if tile.ndim != 2:
            raise RuntimeError(
                f'OME-Zarr tile selector returned shape {tile.shape}; expected one Y/X plane.'
            )
        if remaining_axes == ['X', 'Y']:
            tile = tile.T
        elif remaining_axes != ['Y', 'X']:
            raise RuntimeError(f'Could not normalize remaining axes {remaining_axes} to Y/X.')
        return tile
