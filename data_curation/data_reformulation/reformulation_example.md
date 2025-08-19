## Example Problem 1
Let $a, b, c$ be positive real numbers such that $a + b + c \geq \frac{1}{a} + \frac{1}{b} + \frac{1}{c}$. Find the largest constant $C$ such that the following inequality holds for all $a, b, c$ satisfying the given constraint:
$$
a + b + c \geq \frac{C}{a+b+c} + \frac{2}{abc}.
$$

## Current Solution for Example Problem 1
By $A M \geq H M$ we get
$$
a+b+c \geq \frac{1}{a}+\frac{1}{b}+\frac{1}{c} \geq \frac{9}{a+b+c}
$$
i.e.
$$
\begin{equation*}
\frac{a+b+c}{3} \geq \frac{3}{a+b+c} \tag{1}
\end{equation*}
$$

We will prove that
$$
\begin{equation*}
\frac{2(a+b+c)}{3} \geq \frac{2}{a b c} \tag{2}
\end{equation*}
$$
i.e.
$$
a+b+c \geq \frac{3}{a b c}
$$

Using the well-known inequality $(x y+y z+z x)^{2} \geq 3(x y+y z+z x)$ we obtain
$$
(a+b+c)^{2} \geq\left(\frac{1}{a}+\frac{1}{b}+\frac{1}{c}\right)^{2} \geq 3\left(\frac{1}{a b}+\frac{1}{b c}+\frac{1}{c a}\right)=3 \frac{a+b+c}{a b c}
$$
i.e.
$$
a+b+c \geq \frac{3}{a b c}
$$

After adding (1) and (2) we get the required inequality.

## Expected Answer for Example Problem 1
$C = 3$

## Reformulated Solution for Example Problem 1
By $A M \geq H M$ we get
$$
a+b+c \geq \frac{1}{a}+\frac{1}{b}+\frac{1}{c} \geq \frac{9}{a+b+c}
$$
i.e.
$$
\begin{equation*}
\frac{a+b+c}{3} \geq \frac{3}{a+b+c} \tag{1}
\end{equation*}
$$

We will prove that
$$
\begin{equation*}
\frac{2(a+b+c)}{3} \geq \frac{2}{a b c} \tag{2}
\end{equation*}
$$
i.e.
$$
a+b+c \geq \frac{3}{a b c}
$$

Using the well-known inequality $(x y+y z+z x)^{2} \geq 3(x y+y z+z x)$ we obtain
$$
(a+b+c)^{2} \geq\left(\frac{1}{a}+\frac{1}{b}+\frac{1}{c}\right)^{2} \geq 3\left(\frac{1}{a b}+\frac{1}{b c}+\frac{1}{c a}\right)=3 \frac{a+b+c}{a b c}
$$
i.e.
$$
a+b+c \geq \frac{3}{a b c}
$$

After adding (1) and (2) we get the required inequality.

Equality in the inequalities above holds when $a = b = c = 1$. In this case, $a + b + c = 3$, $\frac{1}{a} + \frac{1}{b} + \frac{1}{c} = 3$, and $abc = 1$, so the original constraint is satisfied with equality, and the inequality becomes $3 \geq \frac{3}{3} + 2 = 3$. Thus, the maximum value of $C$ for which the inequality always holds is $C = 3$.

Therefore, the answer is $C = 3$.


## Example Problem 2
Let $a, b, c \in (-3, 3)$ such that $\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}=\frac{1}{3-a}+\frac{1}{3-b}+\frac{1}{3-c}$. Determine the largest constant $C$ such that the following inequality holds for all $a, b, c$ satisfying the given condition:
$$
\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c} \geq C
$$

## Current Solution for Example Problem 2
By the inequality $A M \geq H M$ we have
$$
\begin{equation*}
((3+a)+(3+b)+(3+c))\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 9 \tag{1}
\end{equation*}
$$
and
$$
\begin{align*}
& ((3-a)+(3-b)+(3-c))\left(\frac{1}{3-a}+\frac{1}{3-b}+\frac{1}{3-c}\right) \geq 9 \\
& \quad \Leftrightarrow \quad((3-a)+(3-b)+(3-c))\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 9 \tag{2}
\end{align*}
$$

After adding (1) and (2) we obtain
$$
18\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 18, \quad \text { i.e. } \quad \frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c} \geq 1
$$

## Expected Answer for Example Problem 2
$C = 1$

## Reformulated Solution for Example Problem 2
By the inequality $A M \geq H M$ we have
$$
\begin{equation*}
((3+a)+(3+b)+(3+c))\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 9 \tag{1}
\end{equation*}
$$
and
$$
\begin{align*}
& ((3-a)+(3-b)+(3-c))\left(\frac{1}{3-a}+\frac{1}{3-b}+\frac{1}{3-c}\right) \geq 9 \\
& \quad \Leftrightarrow \quad((3-a)+(3-b)+(3-c))\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 9 \tag{2}
\end{align*}
$$

After adding (1) and (2) we obtain
$$
18\left(\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c}\right) \geq 18, \quad \text { i.e. } \quad \frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c} \geq 1
$$


Equality holds when $a = b = c = 0$, since then $\frac{1}{3+a}+\frac{1}{3+b}+\frac{1}{3+c} = 3 \times \frac{1}{3} = 1$. This gives the minimum value of the sum, so $C = 1$ is the largest constant for which the inequality always holds.

Therefore, the answer is $C = 1$.
