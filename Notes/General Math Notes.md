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

Formally speaking, instead of writing $+(3,5) = 8$, we write $3+5 = 8$ to denote addition of two numbers. The binary operation $\mu$ is thought of as multiplication and instead of $\mu(a,b)$, we utilize notation such as $ab$, $a + b$, $a \circ b$, and $a\*b$. 

Let $G$ be a finite set of $n$ elements. Then, we can present a binary operation, for example $\*$, by an $n$ by $n$ array called the multiplication table. If $(a,b) \in G$, then the $(a,b)$ entry of this table is $a\*b$.

Let a set, $G = \{a,b,c,d\}$ with the binary operation $*$ acting on it.
Then, we have the following table:

<div align="center">

| $*$ | $a$ | $b$ | $c$ | $d$ |
| --- | --- | --- | --- | --- |
| $a$ | $a$ | $b$ | $c$ | $a$ |
| $b$ | $a$ | $c$ | $d$ | $d$ |
| $c$ | $a$ | $b$ | $d$ | $c$ |
| $d$ | $d$ | $a$ | $c$ | $b$ |
</div>

For this table, note here that $(a\*b)\*c = b\*c = d$, but $a\*(b\*c) = a\*d = a$. This is therefore not a group as a binary operation on a set G must be associative such that $(a\*b)\*c = a\*(b\*c)$.
##### **Lemma 1.2.1**: 
If $(G, \*)$ is a **group** and $a \in G$, then for $a\* a = a$ implies that $a = e$ where $e$ is the identity element. 

<u>Proof</u>: Suppose $a\in G$, then $a\*a = a$ like given. Now suppose $\exists \ b\in G$ such that $b\*a = e$, then it follows that $b\*(a\*a) = b\*a = e$. Therefore, if we have $a$, this is the same thing as an identity element times $a$. Therefore, $a = e\*a = (b\*a)\*a = b\*(a\*a) = b\*a = e$. Therefore, we see that $a = e$.
		
##### **Lemma 1.2.2**:
In a  group, $(G,\*)$, 
 1. if $(b\*a) = e$, then $a\*b = e$ and
 2. $a\*e = a$ for all $a\in G$.

Furthermore, there is one and only one element $e$ in $G$ which satisfies $a\*e = a$. Also, for all $a\in G$, there is only one $b\in G$ which satisfies $b\*a = e$. 

<u>Proof</u>: Suppose $b\*a = e$, then we have 
```math
(a\*b)\*(a\*b) = a\* (b\*a)\*b = a\*e\*b = a\*b.
```

But, from lemma 1.2.1, we saw that $a\*b = e$. Therefore, we have (1). Now, suppose that $a\in G$ and let $b\in G$ such that $b\*a = e$. Then, by (1), we see that 
```math
a\*e = a\*(b\*a) = (a\*b)\*a = e\*a = a, \ \forall a\in G.
```
 
Therefore, we have (2). Now we want to show uniqueness. 


### Tensor Math

##### Tags: #GroupTheory #Math