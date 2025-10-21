### Group Theory

- A group is mathematically defined as a set equipped with a binary operator.
	- A set here is just a collection of functions (i.e different Translations or Rotations)
	- The Group operation operates on elements of this set and this operator tells us how to compose elements of this set.
		- output of this operation must be a **member of the Group**.
	- A group action refers to the group acting on some space, i.e space of pixels.
		- Rotate or transform pixels with the elements in the Group.

```math
\text{Group} \rightarrow (G, \mu) 
```
```math
\mu: \ G \times G \rightarrow G
```

- For the above, for some arbitrary non empty set G, given some binary operation, $\mu$ on G, then $\mu$ is a function such that $G \times G \rightarrow G$. 

Note:
A group needs to fulfill all of the following 4 properties to be considered as a group.
1. Closure - Output never "leaves" the group
2. Associativity - Operation assigned to group is associative
3. Identity - Group contains an identity element which keeps all elements inside the group
4. Inverse - An inverse element exists inside of a group that maps to the identity.

Formally speaking, instead of writing $+(3,5) = 8$, we write $3+5 = 8$ to denote addition of two numbers. The binary operation $\mu$ is thought of as multiplication and instead of $\mu(a,b)$, we utilize notation such as $ab$, $a + b$, $a \circ b$, and $a*b$. 

Let $G$ be a finite set of $n$ elements. Then, we can present a binary operation, for example $*$, by an $n$ by $n$ array called the multiplication table. If $(a,b) \in G$, then the $(a,b)$ entry of this table is $a*b$.

Let a set, $G = \{a,b,c,d\}$ with the binary operation $*$ acting on it.
Then, we have the following table:

| $*$ | $a$ | $b$ | $c$ | $d$ |
| :---: | :---: | :---: | :---: | :---: |
| $a$ | $a$ | $b$ | $c$ | $a$ |
| $b$ | $a$ | $c$ | $d$ | $d$ |
| $c$ | $a$ | $b$ | $d$ | $c$ |
| $d$ | $d$ | $a$ | $c$ | $b$ |


For this table, note here that $(a*b)*c = b*c = d$, but $a*(b*c) = a*d = a$. This is therefore not a group as a binary operation on a set G must be associative such that $(a*b)*c = a*(b*c)$.

### Tensor Math

##### Tags: 