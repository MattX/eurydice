# Eurydice language specification

This specification aims to provide a description of how Everydice should behave. There may be divergences between Everydice and this description - those are bugs, please report them!

This document is intended mostly as a specification for other AnyDice implementers, or for power users. If you're looking for an introduction to the AnyDice DSL, this document is likely too technical. Consider looking at [AnyDice docs](https://anydice.com/docs/) for a smoother introduction.

**This project is not affiliated with AnyDice or Jasper Flick.**

## Overview

Eurydice is a dialect of the [AnyDice](https://anydice.com/) language. It is almost completely compatible with AnyDice, with some extensions, and a few restrictions.

> [!IMPORTANT]
> Intentional divergences from AnyDice are marked with information boxes like this one.

There may also be unintentional divergences between this document and AnyDice's, or Eurydice's, behavior. Please report those on GitHub.

Eurydice is reproducible and hermetic: programs can’t interact with the environment, and two runs will have exactly the same results.

## Lexical elements

A Eurydice program is composed of a sequence of statements. Eurydice is whitespace-insensitive, except to separate tokens.

*Block comments* start and end with backspaces: `\ this is a comment \`.

*Line comments* start with a triple backspace: `\\\ this is a line comment`.

> [!Important]
> Anydice does not support line comments.

Comments are treated as whitespace, and can be used to separate tokens, e.g. `[hello\a comment\world]` is equivalent to `[hello world]`.

*Variable identifiers* are composed of uppercase letters and underscores, for instance `X`, `MY_VARIABLE`, or `_MY_VARIABLE`. `MyVariable`, or `x` are not variable identifiers.

*Words* are composed of lowercase letters and underscores (after the first position only). The single lowercase letter `d` is also not a word. `word`, `if`, `output`, `my_word` are all words. `WORD`, `wOrd`, and `_word` are not.

> [!IMPORTANT]
> AnyDice words (function names) can't contain underscores.

*Punctuation*: the following characters or sequences of characters are tokens:

```
+   =   &   @   :   (   ..
-   !=  |   d   [   )
*   <   !   #   ]   {
/   <=              }
^   >
    >=
```

*Integer literals*: Integer literals are base-10 representations of an unsigned 32-bit integer. It is an error if an integer literal represents a number greater than 2^31-1. Examples include `0`, `123`.

*String literals*: String literal can only occur in a few places. They are enclosed by double quotes (`"`). A string literal can contain escaped double quotes: `"message: \"hello\""`.

## Program execution

A Eurydice program is composed of a sequence of statements. Statements are executed sequentially as specified. Once the program has finished executing, all outputs are displayed to the users.

Errors are fatal and terminate program execution.

## Data types

There are three data types in Eurydice:

* `int`: values of this type hold a 32-bit signed integer
* `list`: values of this type hold a list of 32-bit signed integers.
* `pool`: values of this type hold a pool, which is composed of a mapping of outcomes (each of which is a 32-bit signed integer) to probabilities (whose representation is unspecified), together with an unsigned count of dice, which is called the *dimension*.

The maximum number of elements in a list, or of outcomes in a pool, is 2^31-1.

> [!IMPORTANT]
> AnyDice does not restrict integers, or pool outcomes, to 32-bit values. Experimentation suggests that AnyDice outcomes are represented as double-precision floats (`output 9007199254740993` returns 9007199254740992).

There are no first-class functions.

### Pools

It is not possible to create a pool value representing a pool of different types of dice. For instance, `3d6` is a pool of three d6s, but there is no way to represent a pool of one d6 and one d8.

Pools may have a dimension of 0, or an empty mapping.

The following operations are frequently refered to in this document:

#### Summing

*Summing* a pool transforms it into a pool of dimension 1 whose outcomes are the possible sums when sampling from the dice in the pool, with associated probabilities. Summing a pool of dimension 1 does nothing. Summing a pool of dimension 0 creates a pool of dimension 1 with a single outcome, 0.

For instance, summing `2d2` results in a pool equivalent to `d{2, 3, 3, 4}`.

#### Multiset iteration

*Iterating over the multisets* in a pool refers to processing each possible *sorted* outcome of the collection of dice in the pool, together with its associated probability. For instance, the multisets of `d{1, 2:2, 3:3}` are:

```
multiset              probability
{1, 1, 1}:   (1/6)^3                     * 6
{1, 1, 2}:   (1/6)^2 *  1/3              * 2
{1, 1, 3}:   (1/6)^2           *  1/2    * 2
{1, 2, 2}:    1/6    * (1/3)^2           * 2
{1, 2, 3}:    1/6    *  1/3    *  1/2    * 1
{2, 2, 2}:             (1/3)^3           * 6 
{2, 2, 3}:             (1/3)^2 *  1/2    * 2
{2, 3, 3}:              1/3    * (1/2)^2 * 2
{3, 3, 3}:                       (1/2)^3 * 6
```

If there is a single die in the pool, this is the same as iterating over the die's outcomes.

#### Outcome mapping

*Outcome mapping* a pool transforms each multiset of a pool with a specified operation. If two outcomes are transformed into the same value, their probabilities are summed.

For instance, outcome mapping `2d{-1..1}` with the operation `x→2*abs(x)` results in a pool equivalent to `d{0:3, 2:4, 4:2}`.

#### Flat mapping

*Flat mapping* a pool uses a function that turns an outcome into a pool. The function is called for each multiset of the original pool. The resulting pools are combined together, respecting the probability of the original outcome that produced them.

Each produced pool is summed if its dimension is not 1. The resulting pool always has a dimension of 1.

In pseudocode:

```
def flat_map(pool, f):
    final_outcomes = {}
    for (outcome, prob) in pool.multisets():
        for (new_outcome, new_prob) in f(outcome).sum():
            final_outcomes[new_outcomes] += prob * new_prob
    return final_outcomes
```

For instance, flat mapping `d3` with an operation `x→d(x)` produces a pool equivalent to `d{1:11, 2:5, 3:2}`.

#### Multiset cross product

Starting with pools $p_1, p_2, \ldots, p_n$, the *multiset cross-product* is an iterator over pairs of (list of multiset, probability), where:

* The lists of multisets range over all possible combinations of multisets from each of the pool's multiset iterators
* The probability associated with each list is the product of the multisets' probabilities.

In pseudocode:

```
for (multiset_1, prob_1) in p_1.multisets():
    ...
        for (multiset_n, prob_n) in p_n.multisets():
            yield ([multiset_1, ..., multiset_n], prob_1 * ... * prob_n)
```

## Values, variables, and bindings

All Eurydice *values* are immutable: it is not possible to modify a list or a pool in place.

Eurydice uses [dynamic scoping](https://en.wikipedia.org/wiki/Scope_(computer_science)#Lexical_scope_vs._dynamic_scope). At every point during execution, an *environment* maps variable names to values, and function names to functions. Outside a function, the environment is known as the *global environment*. When execution is about to enter a function's body, a new *environment frame* is pushed to the stack of environments. This environment frame is popped and deleted when execution leaves the function.

```
function: dynamic one {
    X: 1
    result: [x]
}

function: dynamic two {
    X: 2
    result: [x]
}

function: x {
    result: X
}

output [dynamic one]  \\\ Outputs 1
output [dynamic two]  \\\ Outputs 2
```

Values can be bound to variable names in three ways: through an assignment statement, through function parameters, or through a `loop` statement. All methods only affect the innermost environment frame.

Functions can be bound to function names in just one way: with a `function` statement. This method also only affects the innermost frame.

Variable and function bindings are mutable: new values can be assigned to variables, and new functions can be assigned to function names. Only bindings in the innermost environment frame can be mutated. Reusing a name from an outer frame in an assignment will *shadow* the outer binding while the inner frame is live.

```
function x {
    result: X
}

function reassign x {
    X: 2
    result: X
}

X: 0
output [x]           \\\ Outputs 0
X: 1
output [x]           \\\ Outputs 1
output [reassign x]  \\\ Outputs 2
output [x]           \\\ Outputs 1 - the global environment
                     \\\ was not modified by [reassign x].
```

## Global settings

There are three named global settings:

* `explode depth`: set to a non-negative number, default 2. 
* `maximum function depth`: set to a non-negative number, default 10. An attempt to call a function when the current recursion depth is already equal to the maximum function depth will result in the function call not actually executing, and returning an empty list.
* `position order`: set to `"lowest first"` or `"highest first"` (default `"highest first"`). This setting affects two things:
  * the behavior of the [`@` operator](#-operator),
  * the behavior of the [sort function](#sort-SEQUENCEs),
  * the behavior of [calling a function over pools](#Pool-based-evaluation)

Settings can only be set outside of a function, using the [`set` statement](#Set).

## Expressions

### References

The name of a variable is an expression. It evaluates to the value of that variable in the innermost environment frame that has a binding for that variable name.

It is an error to refer to an unbound variable name.

```
X: 3
output X  \\\ Outputs 3
X: {1, 2}
output X  \\\ Outputs 3
X: d4
output X  \\\ Outputs d4
```

```
Y  \\\ Error
```

Note that assigning a pool to a variable *does not* attach the variable to a particular outcome. For instance,

```
X: d4
output X + X   \\\ Output outcomes are {2, 3, 4, 5, 6, 7, 8}
               \\\ *not* {2, 4, 6, 8}.
```

To achieve this effect, use a [pool-evaluated function](#Pool-based-evaluation):

```
function: add D:n to self { result: D + D }
output [add d4 to self]  \\\ Output outcomes: {2, 4, 6, 8}
```

### Literals

The value of an integer literal is the integer it represents.

```
output 25  \\\ Outputs 25
```

### List expressions

A list expression is a comma-separated list of *list elements*, enclosed in curly braces:

```
ListExpr = '{' {ListElem ','} [ListElem [',']] '}'.
```

Once all list elements are evaluated, they are each [flattened into a list](#List-flattening), and concatenated.

#### List elements

Each list element is one of an expression or a range, optionally followed by a repeat count:

```
ListElem = (Expression|Expression '..' Expression) [':' Expression].
```

Expressions composing a list elements are evaluated in an unspecified order. If the list element is a range, both expressions must evaluate to `int` values, and the range is replaced by a list containing all integers between the start and end of the range, inclusive. If the start of the range is greater than the end, an empty list is produced.

If a repeat expression is present, it must evaluate to an `int`. If this value is negative, it is replaced with `0`. The primary expression is then [flattened into a list](#List-flattening), and is concatenated to itself the indicated number of times. If the repeat count is 0, the empty list is produced.

#### List flattening

List flattening transforms values in the following way:

* `int` values are flattened by enclosing them in a single-element list.
* `list` values are unchanged by flattening.
* `pool` values are flattened first by [summing](#Summing), then by discarding the probabilities and creating a list containing each outcome.

#### Examples

```
output {}                \\ Outputs {}
output {1, 2, 3}         \\ Outputs {1, 2, 3}
output {d4}              \\ Outputs {1, 2, 3, 4}
output {d2:2}            \\ Outputs {1, 2, 1, 2}
output {{1, 2}, {3, 4}}  \\ Outputs {1, 2, 3, 4}
```

### Parenthesized expressions

```
'(' Expr ')'.
```

A single expression enclosed in parentheses yields the result of that expression. Explicit parentheses may be used for clarity, or to override the default association of subexpressions.

### Unary operators

There are 4 unary operators, which all bind tighter than any binary operator. There are `!`, `-`, `#`, and `d`.

`!` and `-` compute the logical and arithmetic negation of their argument.

* If the argument is an `int`, `-` negates the value, while `!` evaluates 0 if the argument is nonzero, and 1 otherwise.
* If the argument is a `list`, the values of the list are summed to produce an int.
* If the argument is a `pool`, the pool is summed, then the [outcomes of the pool are all mapped](#Outcome-mapping) with the operator.

`#` evaluates to the length of its argument:

* If the argument is an `int`, it returns the number of digits.
* If the argument is a `list`, it returns the number of elements.
* If the argument is a `pool`, it returns the pool's dimension (count of dice in the pool).

The `d` unary operator acts as the `d` binary operator, with a left argument of 1.

### Binary operators

This is the binary operator precedence list for Eurydice, in decreasing order of precedence:

1. `d`
2. `@`
3. `^`
4. `*`, `/`
5. `+`, `-`
6. `=`, `!=`, `<`, `<=`, `>`, `>=`
7. `&`, `|`

All operators are left-associative.

#### `d` operator

The `d` operator is the main way to create a pool.

1. The RHS operand is converted to a pool.
    1. If it is an `int` `i`, it is converted to a pool with values 1 to `abs(i)` inclusive (if `i` is 0, the pool contains the single outcome 0). If `i` is negative, the pool outcomes then are mapped to their opposite.
    2. If it is a `list`, it is converted to a pool whose outcomes are the distinct values in the list, and whose probability for each outcome is proportional to the number of occurrences of each value in the list.
    3. Pools provided as an RHS operand are not transformed.
2. The LHS operand is summed if it is a `list`, resulting in either an `int` or a `pool`.
    1. If it is an `int` `i`, the count of the RHS pool is multiplied by `abs(i)`. If `i` is negative, then each outcome in the resulting pool is multiplied by `-1`.
    2. If it is a `pool`, then the RHS is [flat mapped](#Flat-mapping) with the operation described in (a).

#### `@` operator

The `@` operator selects the (LHS)-th element from its RHS.

First, if the LHS argument is an int, it is converted to a singleton list. It is an error if the LHS argument is a pool.

* If the RHS is an `int` or `list`, for each value `i` in the resulting list:
  * If the RHS is an `int` `j`, the base-10 digit of `abs(j)` at index `i` is selected. If `j` is negative, this digit is multiplied by -1. If the `"position order"` [global setting](#Global-settings) is set to `"highest first"`, index 1 corresponds to the most significant digit; otherwise, to the least significant. Valid indices start at 1.
  * If the RHS is a `list`, its element at position `i` is selected. The first element of the list has index 1. This is not affected by the `"position order"` setting.
  * In both of these cases, if `i` is invalid (negative or greater than list length), the expression evaluates to 0.
  * Finally, all selected elements are summed to produce an `int`.
* If the RHS argument is a `pool`, then, each outcome multiset is flat mapped with the following function:
  * The multiset is sorted according to the `"position order"` [global setting](#Global-settings).
  * Elements are selected from the multiset and summed as if using the `@` operator from a list.

#### Mathematical operators

The operators `^`, `*`, `/`, `+`, `-`, `&`, and `|` are mathematical operators, performing exponentiation, multiplication, integer division, addition, subtraction, logical conjunction, and logical disjunction, respectively.

Applied to `int`s, `&` evaluates to `1` if both of its arguments are nonzero, `0` otherwise; `|` evaluates to `1` if at least one argument is nonzero. Other mathematical operators have standard behavior on the integers.

1. If either argument to a mathematical operator is a `list`, it is summed to an `int`.
2. After this, if both arguments are `int`s, the operator expression evaluates to an `int`.
3. Otherwise, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS.

#### Comparison operators

The operators `=`, `!=`, `<`, `<=`, `>`, and `>=` are comparison operators, performing equality, inequality, smaller-than, smaller-than-or-equal, larger-than, and larger-than-or-equal checks respectively.

Applied to `int`s, these operators evaluate to `1` if their condition is true, or `0` otherwise.

1. If both arguments are `int`s, the operator expression evaluates to an `int`.
2. If one argument is a `list`, and the other is an `int`, the comparison is performed between the `int` and each member of the `list`. The expression's value is an `int`: the count of comparisons that evaluated to true.
3. If both arguments are `list`s, they are compared in [lexicographic order](https://en.wikipedia.org/wiki/Lexicographic_order).
4. If either argument is a `pool`, both arguments are converted to `pool`s and summed. The LHS is then flat mapped with a function that applies the operator to the LHS and each value in the RHS.

### Function calls

Functions can be called by enclosing the words that make up their names, interleaved with expressions, between square brackets:

```
FunctionCallItem = Word | Expr SafeExpr {SafeExpr} Word.
FinalFunctionCallItem = Expr SafeExpr {SafeExpr}.
FunctionCall = '[' {FunctionCallItem} [FinalFunctionCallItem] ']'.
```

`SafeExpr` is any expression that does not start with `-` or `d`. Because expressions can immediately follow each other, it would otherwise not be possible to know if a call like `[f 1 -2]` should be parsed as `[f (1) (-2)]` or as `[f (1-2)]`.

To evaluate a function call, the function identifier is resolved first. A function is identified by the sequence of words and argument locations in its name, and it is an error if the function's identifier is not in scope in the current environment.

From then, function calls proceed in several steps.

#### Argument coercion

The actual types of the argument expressions are compared to the expected argument types, and each is transformed in the following way:

* If the expected argument type is unspecified, or if the expected type matches the actual type, the value is not transformed.
* If the actual argument is a `list`:
  * If an `int` is requested, the `list` is summed to an `int`.
  * If a `pool` is requested, a `pool` is created, with equally likely outcomes from the `list`.
* If the actual argument type is an `int`:
  * If a `list` is requested, a singleton list is created.
  * if a `pool` is requested, a single-outcome `pool` is created.
* If the actual argument type is a `pool`, and an `int` is requested, the pool is summed, creating a new `pool`.

After this process, some values of type `pool` may still correspond to `int` or `list` type arguments. If this is not the case, the function is called once, and the value of the expression is the result of [evaluating the function](#Function-evaluation).

#### Pool-based evaluation

Otherwise, when there are `pool` values corresponding to `int` or `list` type arguments. Any `pool`s corresponding to an `int` argument is summed to dimension 1. Then, the [multiset cross product](#Multiset-cross-product) of the pools is generated. The order in which elements inside each multiset are ordered follows the `"position order"` [global setting](#Global-settings).

The function is then evaluated once for each value in the multiset cross product (this may be 0 times if the cross product is empty). In each invocation, the argument values are:

* For any argument whose actual type correspond to its declared type, the actual value
* For arguments where a `pool` was provided but a `list` or `int` was requested, that pool's multiset value in the current element of the multiset cross product iterator.

All results are collected then collected, together with the probability associated with the list of multisets that produced it.

#### Function evaluation

A [new environment frame](#Values-variables-and-bindings) is created, binding the value of each expression to its corresponding argument name. Each statement in the function's body is then evaluated sequentially. If a [return statement](#Return-from-function) is encountered, execution returns to the caller, and the function call's value is the return statement's expression.

If no return statement is encountered, the function returns an empty list.

## Statements

### Conditionals

Statements can be executed conditionally:

```
IfStatement = 'if' Expr Block ['else' (IfStatement | Block)].
```

The condition expression is evaluated. It is an error if it does not evaluate to an `int`. If it evaluates to `0`, the `else` block, if any, is executed statement by statement. Otherwise, the `if` block is executed.

There is optional special syntax if the `else` block is itself an `if` statement.

### Loops

A loop executes a body repeatedly while changing the value of a binding:

```
LoopStatement = `loop` VariableName `over` Expr Block.
```

The loop expression is evaluated. It is an error if it does not evaluate to a list. A binding for `VariableName` is created in the current environment. For each value in the list, the loop variable is bound to that value, then all statements in the body are executed in succession.

If there are no values in the list, the loop does not execute.

### Print and output

Both statements allow the user to output a value.

```
PrintStatement = 'print' Expr ['named' Str].
OutputStatement = 'output' Expr ['named' Str].
```

> [!IMPORTANT]
> AnyDice does not support the `print` statement.

It is an error if an `output` statement occurs inside a function. No such restriction exists for the `print` statement.

In both cases, the expression is evaluated. If present, the name string is then interpolated: any pair of square brackets containing a *variable name* will be replaced by the named variable's value. Square brackets containing any other data will not be replaced.

It is an error for an undefined variable name to occur within square brackets.

> [!IMPORTANT]
> Eurydice is a little more explicit than AnyDice in replaced variables for lists and pools, using strings like `{1, 3, 4}` instead of `{?}`.

For an `output` statement, the value of the expression is converted to a `pool`, and added to an output list. If a name is not provided, it is associated with the default name `output n`, where `n` is the 1-indexed output number.

For a `print` statement, the value of the expression is shown to the user as soon as possible, attached to the name if present. There is no default name otherwise.

### Set

The `set` statement allows changing [global settings](#Global-settings). It is an error for a `set` statement to occur inside a function.

```
SetStatement = 'set' '"position order"' 'to' '"highest first"'.
SetStatement = 'set' '"position order"' 'to' '"lowest first"'.
SetStatement = 'set' '"explode depth"' 'to' Int.
SetStatement = 'set' '"maximum function depth"' 'to' Int.
```

> [!IMPORTANT]
> Eurydice does not allow arbitrary expressions in set statements. Arguments must be int literals.

### Assignment

Assignment statements either create a new binding in the [innermost environment frame](#Values-variables-and-bindings), or replace the value of an existing binding, if a variable with the same name already exists in that frame.

```
AssignmentStatement = VariableName ':' Expr.
```

### Function definition

Function definitions create a new function binding in the [innermost environment frame](#Values-variables-and-bindings), or replace the value of an existing binding, if a function with the same identifier already exists in that frame.

```
FunctionDefinitionStatement = 'function' ':' (Word | Parameter)+ Block.
Parameter: ParameterName [':' Type].
```

> [!IMPORTANT]
> AnyDice function definitions can only occur at the top level.

The function's identifier is the sequence of words and argument positions in the name. It is valid for a function identifier to contain no words, or to contain no argument positions.

Each argument name can optionally be annotated with a type.

### Return from function

The `result` statement allows returning a value from a function. It is an error for `result` to appear outside a function.

```
ResultStatement = 'result' ':' Expr.
```

Control flow immediately returns to the current function's caller.

## Built-in functions

The following functions are available in the top-level environment frame at the beginning of a program:

### `[absolute N:n]`

Returns the absolute value of `n`.

### `[SEQ:s contains N:n]`

Returns `1` if `SEQ` contains `N`, or `0` otherwise.

### `[count NEEDLES:s in HAYSTACK:s]`

Returns the number of occurrences of each element of `NEEDLES` in `HAYSTACK`.

Examples:

```
output [count {1, 2} in {3, 4}]  \ Outputs 0 \
output [count {1, 1, 2} in {1, 2, 2, 3}]  \ Outputs 4 \
```

### `[explode POOL:d]`

This transforms a die to match an [explosion rule](https://nethackwiki.com/wiki/Exploding_die): if the die rolls its highest face value, that value is kept and the die is re-rolled.

The maximum number of rerolls is controlled by the `"explode depth"` [global setting](#Global-settings).

If the argument is a pool with dimension >= 1, it is summed first. The return value's dimension is always 0 or 1.

```
output [explode d{}]  \ Outputs d{} \

\ Outputs d{1:9, 2:9, 4:3, 5:3, 7, 8, 9} \
output [explode d{1, 2, 3}]
```

### `[highest COUNT:n of POOL:d]`, `[lowest COUNT:n of POOL:d]`, `[middle COUNT:n of POOL:d]`

These functions return pools of dimension 1 whose distribution the sum of the highest `n`, lowest `n`, or middle `n` values in the possible outcomes of `POOL`.

`[middle COUNT:n of POOL:d]` rounds indices below. For instance, `[middle 1 of 4d2]` will return the second-lowest value for each multiset (index `(4-1)/2=1`, 0-based).

### `[highest of FIRST:n and SECOND:n]`, `[lowest of FIRST:n and SECOND:n]`

These functions return the max and min of their two arguments, respectively.

### `[maximum POOL:d]`

Sums `POOL`, then returns its largest possible outcome.

### `[reverse SEQUENCE:s]`

Returns `SEQUENCE` in reverse order.

### `[sort SEQUENCE:s]`

This function returns a sequence with the same elements as `SEQUENCE`, sorted in ascending or descending order depending on the value of the `"position order"` global setting.

## Acknowledgements

The Eurydice language is almost entirely identical to [AnyDice by Jasper Flick](https://anydice.com/). Thanks!

This spec is inspired by the [Starlark spec](https://github.com/bazelbuild/starlark/blob/master/spec.md).
