# Finite-difference discretization

The FDM code assembles a diffusion stiffness matrix `K`, with a positive
diagonal, in tissue indexing. Time integrators use it with the sign required by
the diffusion equation, for example `I - dt * K` in the explicit update.

## Mesh and indexing

`mesh` uses three cell states:

- `0`: outside the tissue and absent from the matrices;
- `1`: active myocyte;
- `2`: tissue state retained in the matrices but decoupled from diffusion (e.g fibrosis).

Matrix rows and columns follow the compressed array `mesh[mesh > 0]`.
Consequently, `indexes` passed to `compute_diffusion_operator` are positions in
that compressed array, not flat indexes of the full grid. If `indexes` is
omitted, every `mesh == 1` cell is assembled.

## Coefficient layouts

Diffusion can be supplied as:

- a scalar isotropic coefficient;
- one constant `(ndim, ndim)` tensor;
- a `mesh.shape + (ndim, ndim)` full-grid field;
- a `(tissue_size, ndim, ndim)` compressed field.

Connectivity describes oriented positive-axis edges. `connectivity[..., axis]`
belongs to the edge from a cell to its `+1` neighbor on that axis. It can be a
scalar, an `(ndim,)` vector, a `mesh.shape + (ndim,)` field, or a compressed
`(tissue_size, ndim)` field.

Cell-centered diffusion tensors are averaged at an edge. Arithmetic averaging
is the default; harmonic averaging is available through
`AsymmetricDiscretization("harmonic")`.

## Discretizations

`AsymmetricDiscretization` assembles each positive-axis face once. The same
linear face-flux expression is added to the center row and subtracted from the
positive-neighbor row. This makes the scheme locally conservative: a flux that
leaves one cell enters the other.

For a face normal to axis `a`, the tensor row `D[a, :]` is interpolated from
the two adjacent cell centers. Its diagonal component `D[a, a]` multiplies the
normal two-point difference. Every off-diagonal component `D[a, b]` uses four
cells surrounding the face to approximate the transverse derivative along
axis `b`. Therefore a 2D tensor normally produces a nine-point stencil and a
3D tensor can involve additional edge-diagonal neighbors.

Boundary handling is deliberately local to a face:

- if the positive major neighbor is invalid, the whole face flux is zero;
- if any of the four cells needed by one transverse derivative is invalid,
  only that transverse term is zero;
- unlike `IsotropicDiscretization`, no opposite neighbor is mirrored onto a
  missing face.

The scheme preserves constants: every assembled row sums to zero. However,
for nonzero off-diagonal tensor components the resulting matrix is generally
**not symmetric**, even when the continuous diffusion tensor is symmetric.
Flux conservation does not imply matrix symmetry. Consequently, iterative
solvers that require a symmetric positive-definite matrix, including
Conjugate Gradient, are not mathematically guaranteed for this fully
anisotropic operator. With scalar diffusion or an axis-aligned diagonal tensor,
the mixed terms disappear and the usual symmetric centered interior stencil is
recovered.

`IsotropicDiscretization` uses only diagonal tensor components. At a no-flux
boundary it mirrors the valid opposite neighbor, producing the second-order
one-sided stencil. If both neighbors on an axis are invalid, that axis adds no
flux.

For both methods, every assembled row sums to zero, so a constant field is
preserved. A `mesh == 2` row and column remain zero in `K`; the corresponding
identity entry remains present in the FDM mass matrix.

## Assembly outline for `AsymmetricDiscretization`

For each axis and each active center point:

1. `build_neighbor` forms the positive major-neighbor coordinates.
2. `is_valid_index` marks faces whose two main cells are active (`mesh == 1`).
3. `_average_diffusion_component` interpolates the relevant tensor row to the
   face; `_connectivity_component` reads the multiplier stored at the center's
   positive edge.
4. `_major_flux_weights` creates the normal two-point contribution.
5. `_minor_flux_weights` creates one four-point transverse contribution per
   remaining axis when its complete support is valid.
6. `nonzero_weights(..., direction=1)` emits the face expression into the
   center row, while `direction=-1` emits its negative into the major row.
7. The component method divides by `dr` for the discrete divergence. Flux
   weights already contain `1 / dr` from the gradient, giving the expected
   overall `1 / dr**2` scaling.
8. SciPy sums duplicate COO entries while constructing the final CSR matrix.
