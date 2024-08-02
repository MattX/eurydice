# Everydice documentation

## Examples

```
2d6 + 1
```

```
d{-1..1}
```

```
sum(drop_lowest(3d6, 1))
```

```
coin <- d2
if coin == 1 { d4 } else { 21-d4 }
```

```
highest_and_sum = [0, 0]
for i in [1..20] {
    highest_and_sum = do {
        roll <- d6
        [highest, sum] <- highest_and_sum
        [max(roll, highest), roll + sum]
    }
}
highest_and_sum
```

## Types

There are essentially 4 types of values:

* `int` is a plain integer
* `list` is a list of integers
* `dist` is a probability distribution over lists of integers. The list can be one integer long. To save space, distributions are internally represented as lists of distribitions, where each component of the list is independent of the others.<br/>For instance, `2d6` has runtime type `dist[1, 1]`, indicating that it is a distribution over lists of length 2 made up of two independent components. Lists can be `dist[?]`, but these are less useful.
* Functions are first-class values as well.

### Subtyping

Everydice has a notion of subtyping: `int <: list <: dist`.

Lists of lists and lists of dists do not exist. Attempting to place lists in lists will result in the lists being concatenated, for instance:

`[[1, 2, 3], 4, [5, 6, 7]]`

evaluates to

`[1, 2, 3, 4, 5, 6, 7]`.

Similarly, putting a distribution in a list will result in the whole list becoming a distribution: `[1, 1d6, [4, 5, 6]]` has type `dist[1, 1, 3]`.

## Syntax

It tries, and fails, to be close to Python.

## Standard library

### Unary operators

* `!: int -> int; list -> list; dist -> dist`: elementwise logical negation
* `-: int -> int; list -> list; dist -> dist`: elementwise arithmetic negation

### Binary operators

* `+`, `*`:
    * `(int, int) -> int`: arithmetic addition and multiplication
    * `(list, list) -> list`, `(dist, dist) -> dist`: perform the total sum (for `+`) or product (for `*`) of both distributions. Broadcasts one side if necessary.
* `-`
    * `(int, int) -> int`: subtraction
    * `(list, list) -> list` or `(dist, dist) -> dist`: performs `sum(left) - sum(right)`.
* `/`: `(int, int) -> int`: integer division rounding down
* `==`, `!=`, `<=`, `<`, `>`, `>=`:
    * `(int, int) -> int`: logical comparisons
    * On lists or distributions, applies elementwise. Both sides must have the same dimension.
* `or`, `and`:
    * `(int, int) -> int`: returns 0 if both (resp. either) sides are 1, 0 otherwise.
    * On lists or distributions, applies elementwise. Both sides must have the same dimension.

### Built-in functions

On dists:

* `cross(d: dist) -> dist`: performs the full product of the distributions in `d`, turning a `dist[x, y, z, ...]` into a `dist[x+y+z]`. Potentially very expensive.
* `marginalize(d: dist) -> dist`: the opposite of `cross`. Loses all covariance information.
* `sum(d: dist) -> dist`, `product(d: dist) -> dist`: computes the sum and product, turning the dist into a `dist[1]`.
* `drop_low(d: dist, n: int) -> dist`, `drop_high(d: dist, n: int)`, `keep_low(d: dist, n: int) -> dist`, `keep_high(d: dist, n: int)`: drop / keep the n lowest or highest values.
* `dim(d: dist) -> list`: returns a list containing the sizes of indepdent sub-distributions.
* `size(d: dist) -> int`: returns the cardinality of the dist. Equivalent to `sum(dim(d))`.
* `map(f: function[list -> dist], d: dist)`: applies function `f` to each outcome in `d`, producing a new distribution.

On dists or lists:

* `get(d: dist, n: int) -> dist`: get the nth element of a dist. The result has type `dist[1]`.
* `get(d: list, n: int) -> int`: gets the nth element of a list.
