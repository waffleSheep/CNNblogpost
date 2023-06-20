# Building Convolutional Neural Networks from scratch by Karimi Zayan

## Part 2: Convolution and Max Pooling

Hello, welcome to Part 2. If you haven't read part 1, I strongly suggest you read it to gain a deeper understanding about the intuition about matrices more properly. This time, we are ending off the 2 part series with Convolution and Max Pooling Math. By the end of this, I will use all the knowledge over the 2 parts and build a network to fit a small image dataset.

## Things to note

### Changing and swapping notation

In the previous article, I stuck to the numerator layout. However, for this article, we may use the denominator layout for convenience as it makes more sense. However, usually, the inbetween on such steps are flat layers and have various ravel operations futher obfuscating the divide between the 2 competing notations

### Tensor Analysis

I have decided to avoid tensor analysis because it is very difficult and horrible. It is more horrible than the 1000 derivatives you wil see below. In the appendix section, I attempt to do this with einstein summation.

### Non-rigorous Non-proofs

The individual derivatives are for small examples and are non-rigorous and are not proofs that these hold for any size of matrix or vector. This has made some people mad but like the proofs would be so simple, I do not know why yall are so angry. Anyway, feel free to email obsceneties to my second personal email prannayagupta@gmail.com

### Implementation

Unlike with Standard DNN layers, implementation of the layers is non-trivial so some chunk of this article will be spent disecting that.

### It has been a year!!!

Actually, it has been more than a year. I have been doing a lot of random other stuff and the focus has always been shifting but hey, it is out and free now!

## Recap on the last blogpost

Since we are shifting notation to the denomenator layout, I will recap and also rewrite the derivatives we calculated last time. I will also make it more compact than last time since writing the full thing out may be quite confused.

The equations for Feed Forward are

$$
\begin{aligned}
\mathbf{a}_0 &= \mathbf{x} \\
\mathbf{z}_{i+1} &= W_i \mathbf{a}_i+\mathbf{b}_i \\
\mathbf{a}_i &= \sigma (\mathbf{z}_i) \\
c &= \frac{1}{n}||\mathbf{a}_L-\mathbf{y}||^2_2 \\
\end{aligned}
$$

The derivative for Mean-Squared Error is

$$
\frac{\partial c}{\partial \mathbf{a}_L} = \frac{2}{n}(\mathbf{a}_L-\mathbf{y})
$$

The derivative for Element-Wise Activation is

$$
\frac{\partial c}{\partial \mathbf{z}_i} =  \text{diag}(\sigma^{'}(\mathbf{z}_i)) \frac{\partial c}{\partial \mathbf{a}_i}
$$

The derivative for Linear Layer is

$$
\begin{aligned}
\frac{\partial c}{\partial \mathbf{a}_i} &= W_i^T \frac{\partial c}{\partial \mathbf{z}_{i+1}}\\
\frac{\partial c}{\partial \mathbf{b}_i} &=\frac{\partial c}{\partial \mathbf{z}_{i+1}}\\
\frac{\partial c}{\partial W_i} &= \frac{\partial c}{\partial \mathbf{z}_{i+1}}\mathbf{a}_i^T
\end{aligned}
$$

## Convolution (Single Layer)

Mathematical convolution is defined as

$$
(f*g)(t):=\int _{-\infty }^{\infty }f(\tau )g(t-\tau )\,d\tau
$$

However, this definition is only for functions which has a domain  $\mathbb{R}$. Essentially, convolution is kind of like sliding window multiplication. Convolution works better than a standard deep neural network layer for image because every feature is not analysed pixel wise but a bunch of pixels together are analysed (local spatial coherence). There is also an order to the inputs, not seen by Deep Neural Network layers. I will not elaborate any further because this article is about the math and not about the rationale. So let us continue. 2D convolution of 2 matrices is defined as such

$$
\begin{aligned}

W * X &= 

\begin{bmatrix}

{w}_{11} & {w}_{12} & {w}_{13}\\

{w}_{21} & {w}_{22} & {w}_{23}\\

{w}_{31} & {w}_{32} & {w}_{33}\\

\end{bmatrix}*

\begin{bmatrix}

{x}_{11} & {x}_{12}\\

{x}_{21} & {x}_{22}\\

\end{bmatrix}\\

&=

\begin{bmatrix}

{w}_{11}{x}_{11}+{w}_{12}{x}_{12}+{w}_{21}{x}_{21}+{w}_{22}{x}_{22} & {w}_{12}{x}_{11}+{w}_{13}{x}_{12}+{w}_{22}{x}_{21}+{w}_{23}{x}_{22}\\

{w}_{21}{x}_{11}+{w}_{22}{x}_{12}+{w}_{31}{x}_{21}+{w}_{32}{x}_{22} & {w}_{22}{x}_{11}+{w}_{23}{x}_{12}+{w}_{32}{x}_{21}+{w}_{33}{x}_{22}\\

\end{bmatrix}\\

\end{aligned}

$$

It is similar to sliding window multiplication. Since this function is $f:\mathbb{R}^{n \times m} \rightarrow \mathbb{R}^{p \times q}$, the derivative will be a fourth order tensor. So it is time, to look at the overall equations.

$$
\begin{aligned}

Y&=W*X\\

\begin{bmatrix}

{y}_{11} & {y}_{12}\\

{y}_{21} & {y}_{22}\\

\end{bmatrix}&=

\begin{bmatrix}

{w}_{11} & {w}_{12} & {w}_{13}\\

{w}_{21} & {w}_{22} & {w}_{23}\\

{w}_{31} & {w}_{32} & {w}_{33}\\

\end{bmatrix}*

\begin{bmatrix}

{x}_{11} & {x}_{12}\\

{x}_{21} & {x}_{22}\\

\end{bmatrix}\\&=

\begin{bmatrix}

{w}_{11}{x}_{11}+{w}_{12}{x}_{12}+{w}_{21}{x}_{21}+{w}_{22}{x}_{22} & {w}_{12}{x}_{11}+{w}_{13}{x}_{12}+{w}_{22}{x}_{21}+{w}_{23}{x}_{22}\\

{w}_{21}{x}_{11}+{w}_{22}{x}_{12}+{w}_{31}{x}_{21}+{w}_{32}{x}_{22} & {w}_{22}{x}_{11}+{w}_{23}{x}_{12}+{w}_{32}{x}_{21}+{w}_{33}{x}_{22}\\

\end{bmatrix}\\

\end{aligned}

$$

We have to task find the cost with respect to each weight and the cost with respect to each layer. Let us start with $\frac{\partial Y}{\partial X}$.

$$
\frac{\partial {y}_{11}}{\partial {x}_{11}} = {w}_{11} \quad

\frac{\partial {y}_{11}}{\partial {x}_{12}} = {w}_{12} \quad

\frac{\partial {y}_{11}}{\partial {x}_{21}} = {w}_{21} \quad

\frac{\partial {y}_{11}}{\partial {x}_{12}} = {w}_{22} \\

\frac{\partial {y}_{12}}{\partial {x}_{11}} = {w}_{12} \quad

\frac{\partial {y}_{12}}{\partial {x}_{12}} = {w}_{13} \quad

\frac{\partial {y}_{12}}{\partial {x}_{21}} = {w}_{22} \quad

\frac{\partial {y}_{12}}{\partial {x}_{12}} = {w}_{23} \\

\frac{\partial {y}_{21}}{\partial {x}_{11}} = {w}_{21} \quad

\frac{\partial {y}_{21}}{\partial {x}_{12}} = {w}_{22} \quad

\frac{\partial {y}_{21}}{\partial {x}_{21}} = {w}_{31} \quad

\frac{\partial {y}_{21}}{\partial {x}_{12}} = {w}_{32} \\

\frac{\partial {y}_{22}}{\partial {x}_{11}} = {w}_{22} \quad

\frac{\partial {y}_{22}}{\partial {x}_{12}} = {w}_{23} \quad

\frac{\partial {y}_{22}}{\partial {x}_{21}} = {w}_{32} \quad

\frac{\partial {y}_{22}}{\partial {x}_{12}} = {w}_{33} \\

$$

Once again, because at the end of the day, everything gets compressed down to one scalar quantity, we use our special function $f:\mathbb{R}^{n \times m} \rightarrow \mathbb{R}$, but with a slight modification. This time, the function goes from a matrix to a scalar. It is defined as $c = f(Y)$. It refers to an arbritary matrix to scalar function. So let us work out the general form for the derivative $\frac{\partial c}{\partial X}$ by considering $\frac{\partial c}{\partial Y}$ first. In this case, we will use denominator layout for simplicity.

$$
\begin{aligned}
\frac{\partial c}{\partial Y}&=
\begin{bmatrix}

\frac{\partial c}{\partial {y}_{11}} & \frac{\partial c}{\partial {y}_{12}}\\

\frac{\partial c}{\partial {y}_{21}} & \frac{\partial c}{\partial {y}_{22}}\\

\end{bmatrix}\\
\frac{\partial c}{\partial X}&=
\begin{bmatrix}

\frac{\partial c}{\partial {x}_{11}} & \frac{\partial c}{\partial {x}_{12}}\\

\frac{\partial c}{\partial {x}_{21}} & \frac{\partial c}{\partial {x}_{22}}\\

\end{bmatrix}\\&=
\begin{bmatrix}

\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {x}_{11}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {x}_{11}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {x}_{11}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {x}_{11}} & \frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {x}_{12}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {x}_{12}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {x}_{12}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {x}_{12}}\\

\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {x}_{21}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {x}_{21}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {x}_{21}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {x}_{21}} & \frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {x}_{22}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {x}_{22}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {x}_{22}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {x}_{22}}\\

\end{bmatrix}\\&=

\begin{bmatrix}

\frac{\partial c}{\partial {y}_{11}}{w}_{11}+\frac{\partial c}{\partial {y}_{12}}{w}_{12}+\frac{\partial c}{\partial {y}_{21}}{w}_{21}+\frac{\partial c}{\partial {y}_{22}}{w}_{22} & \frac{\partial c}{\partial {y}_{11}}{w}_{12}+\frac{\partial c}{\partial {y}_{12}}{w}_{13}+\frac{\partial c}{\partial {y}_{21}}{w}_{22}+\frac{\partial c}{\partial {y}_{22}}{w}_{23}\\

\frac{\partial c}{\partial {y}_{11}}{w}_{21}+\frac{\partial c}{\partial {y}_{12}}{w}_{22}+\frac{\partial c}{\partial {y}_{21}}{w}_{31}+\frac{\partial c}{\partial {y}_{22}}{w}_{32} & \frac{\partial c}{\partial {y}_{11}}{w}_{22}+\frac{\partial c}{\partial {y}_{12}}{w}_{23}+\frac{\partial c}{\partial {y}_{21}}{w}_{32}+\frac{\partial c}{\partial {y}_{22}}{w}_{33}\\
\end{bmatrix}\\&=

\begin{bmatrix}

{w}_{11} & {w}_{12} & {w}_{13}\\

{w}_{21} & {w}_{22} & {w}_{23}\\

{w}_{31} & {w}_{32} & {w}_{33}\\

\end{bmatrix}*
\begin{bmatrix}

\frac{\partial c}{\partial {y}_{11}} & \frac{\partial c}{\partial {y}_{12}}\\

\frac{\partial c}{\partial {y}_{21}} & \frac{\partial c}{\partial {y}_{22}}\\

\end{bmatrix}\\&=
W*\frac{\partial c}{\partial Y}
\end{aligned}
$$

Now let us move onto $\frac{\partial Y}{\partial W}$ by beginning de novo with the individual derivatives. There are 36 of them. While that is a lot, it is the most, we will handle in this article.

$$
\frac{\partial {y}_{11}}{\partial {w}_{11}} = {x}_{11} \quad

\frac{\partial {y}_{12}}{\partial {w}_{11}} = 0 \quad

\frac{\partial {y}_{21}}{\partial {w}_{11}} = 0 \quad

\frac{\partial {y}_{22}}{\partial {w}_{11}} = 0 \\

\frac{\partial {y}_{11}}{\partial {w}_{12}} = {x}_{12} \quad

\frac{\partial {y}_{12}}{\partial {w}_{12}} = {x}_{11} \quad

\frac{\partial {y}_{21}}{\partial {w}_{12}} = 0 \quad

\frac{\partial {y}_{22}}{\partial {w}_{12}} = 0 \\

\frac{\partial {y}_{11}}{\partial {w}_{13}} = 0 \quad

\frac{\partial {y}_{12}}{\partial {w}_{13}} = {x}_{12} \quad

\frac{\partial {y}_{21}}{\partial {w}_{13}} = 0 \quad

\frac{\partial {y}_{22}}{\partial {w}_{13}} = 0 \\

\frac{\partial {y}_{11}}{\partial {w}_{21}} = {x}_{21} \quad

\frac{\partial {y}_{12}}{\partial {w}_{21}} = 0 \quad

\frac{\partial {y}_{21}}{\partial {w}_{21}} = {x}_{11} \quad

\frac{\partial {y}_{22}}{\partial {w}_{21}} = 0 \\

\frac{\partial {y}_{11}}{\partial {w}_{22}} = {x}_{22} \quad

\frac{\partial {y}_{12}}{\partial {w}_{22}} = {x}_{21} \quad

\frac{\partial {y}_{21}}{\partial {w}_{22}} = {x}_{12} \quad

\frac{\partial {y}_{22}}{\partial {w}_{22}} = {x}_{11} \\

\frac{\partial {y}_{11}}{\partial {w}_{23}} = 0 \quad

\frac{\partial {y}_{12}}{\partial {w}_{23}} = {x}_{22} \quad

\frac{\partial {y}_{21}}{\partial {w}_{23}} = 0 \quad

\frac{\partial {y}_{22}}{\partial {w}_{23}} = {x}_{12} \\

\frac{\partial {y}_{11}}{\partial {w}_{31}} = 0 \quad

\frac{\partial {y}_{12}}{\partial {w}_{31}} = 0 \quad

\frac{\partial {y}_{21}}{\partial {w}_{31}} = {x}_{21} \quad

\frac{\partial {y}_{22}}{\partial {w}_{31}} = 0 \\

\frac{\partial {y}_{11}}{\partial {w}_{32}} = 0 \quad

\frac{\partial {y}_{12}}{\partial {w}_{32}} = 0 \quad

\frac{\partial {y}_{21}}{\partial {w}_{32}} = {x}_{22} \quad

\frac{\partial {y}_{22}}{\partial {w}_{32}} = {x}_{21} \\

\frac{\partial {y}_{11}}{\partial {w}_{33}} = 0 \quad

\frac{\partial {y}_{12}}{\partial {w}_{33}} = 0 \quad

\frac{\partial {y}_{21}}{\partial {w}_{33}} = 0 \quad

\frac{\partial {y}_{22}}{\partial {w}_{33}} = {x}_{22} \\
$$

We will now find $\frac{\partial c}{\partial W}$

$$
\begin{aligned}
\frac{\partial c}{\partial {w}_{11}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{11}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{11}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{11}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{11}}\\
&=\frac{\partial c}{\partial {y}_{11}}{x}_{11}\\

\frac{\partial c}{\partial {w}_{12}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{12}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{12}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{12}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{12}}\\
&=\frac{\partial c}{\partial {y}_{11}}{x}_{12}+\frac{\partial c}{\partial {y}_{12}}{x}_{11}\\

\frac{\partial c}{\partial {w}_{13}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{13}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{13}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{13}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{13}}\\
&=\frac{\partial c}{\partial {y}_{12}}{x}_{12}\\

\frac{\partial c}{\partial {w}_{21}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{21}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{21}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{21}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{21}}\\
&=\frac{\partial c}{\partial {y}_{11}}{x}_{21}+\frac{\partial c}{\partial {y}_{21}}{x}_{11}\\

\frac{\partial c}{\partial {w}_{22}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{22}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{22}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{22}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{22}}\\
&=\frac{\partial c}{\partial {y}_{11}}{x}_{22}+\frac{\partial c}{\partial {y}_{12}}{x}_{21}+\frac{\partial c}{\partial {y}_{21}}{x}_{12}+\frac{\partial c}{\partial {y}_{22}}{x}_{11}\\

\frac{\partial c}{\partial {w}_{23}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{23}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{23}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{23}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{23}}\\
&=\frac{\partial c}{\partial {y}_{12}}{x}_{22}+\frac{\partial c}{\partial {y}_{22}}{x}_{12}\\

\frac{\partial c}{\partial {w}_{31}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{31}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{31}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{31}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{31}}\\
&=\frac{\partial c}{\partial {y}_{21}}{x}_{21}\\

\frac{\partial c}{\partial {w}_{32}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{32}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{32}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{32}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{32}}\\
&=\frac{\partial c}{\partial {y}_{21}}{x}_{22}+\frac{\partial c}{\partial {y}_{22}}{x}_{21}\\

\frac{\partial c}{\partial {w}_{33}}&=\frac{\partial c}{\partial {y}_{11}}\frac{\partial {y}_{11}}{\partial {w}_{33}}+\frac{\partial c}{\partial {y}_{12}}\frac{\partial {y}_{12}}{\partial {w}_{33}}+\frac{\partial c}{\partial {y}_{21}}\frac{\partial {y}_{21}}{\partial {w}_{33}}+\frac{\partial c}{\partial {y}_{22}}\frac{\partial {y}_{22}}{\partial {w}_{33}}\\
&=\frac{\partial c}{\partial {y}_{22}}{x}_{22}\\
\end{aligned}
$$

Now we organize all the data in a matrix

$$
\begin{aligned}
\frac{\partial c}{\partial {W}}&=

\begin{bmatrix}

\frac{\partial c}{\partial {w}_{11}} & \frac{\partial c}{\partial {w}_{12}} & \frac{\partial c}{\partial {w}_{13}}\\

\frac{\partial c}{\partial {w}_{21}} & \frac{\partial c}{\partial {w}_{22}} & \frac{\partial c}{\partial {w}_{23}}\\

\frac{\partial c}{\partial {w}_{31}} & \frac{\partial c}{\partial {w}_{32}} & \frac{\partial c}{\partial {w}_{33}}\\

\end{bmatrix}\\
&=
\begin{bmatrix}

\frac{\partial c}{\partial {y}_{11}}{x}_{11} & \frac{\partial c}{\partial {y}_{11}}{x}_{12}+\frac{\partial c}{\partial {y}_{12}}{x}_{11} & \frac{\partial c}{\partial {y}_{12}}{x}_{12}\\

\frac{\partial c}{\partial {y}_{11}}{x}_{21}+\frac{\partial c}{\partial {y}_{21}}{x}_{11} & \frac{\partial c}{\partial {y}_{11}}{x}_{22}+\frac{\partial c}{\partial {y}_{12}}{x}_{21}+\frac{\partial c}{\partial {y}_{21}}{x}_{12}+\frac{\partial c}{\partial {y}_{22}}{x}_{11} & \frac{\partial c}{\partial {y}_{12}}{x}_{22}+\frac{\partial c}{\partial {y}_{22}}{x}_{12}\\

\frac{\partial c}{\partial {y}_{21}}{x}_{21} & \frac{\partial c}{\partial {y}_{21}}{x}_{22}+\frac{\partial c}{\partial {y}_{22}}{x}_{21} & \frac{\partial c}{\partial {y}_{22}}{x}_{22}\\

\end{bmatrix}\\

&=
\begin{bmatrix}

0 & 0 & 0 & 0\\
0 & \frac{\partial c}{\partial {y}_{11}} & \frac{\partial c}{\partial {y}_{12}} & 0\\

0 & \frac{\partial c}{\partial {y}_{21}} & \frac{\partial c}{\partial {y}_{22}} & 0\\

0 & 0 & 0 & 0\\

\end{bmatrix}*
\begin{bmatrix}

{x}_{22} & {x}_{21}\\

{x}_{12} & {x}_{11}\\

\end{bmatrix}\\
&=\text{pad}\left(\frac{\partial c}{\partial {Y}}\right)*\text{flip}(X)

\end{aligned}
$$

Voila, after that pain, we have done it. In this case $\text{pad}$ refers to adding zeroes all around and $\text{flip}$ refers to a flip of the matrix in both the horizontal and vertical axes. The amount of $\text{pad}$ is one less than the size of the kernel.

## Max-pooling (Single Layer)

Now we will find the derivative of a max-pooling layer. Any sort of operation which involves the movement of numbers around has rather simple solution when it comes to the propogation of derivatives backwards. Before we think of max-pooling let us just think of a simple max function $\text{max}:\mathbb{R}^{n} \rightarrow \mathbb{R}$. We feed a simple vector into this function

$$
\begin{aligned}
\mathbf{x}&=\begin{bmatrix}
5 \\
9 \\
7 \\
6 \\
\end{bmatrix}\\
y&=max \left( \begin{bmatrix}
5 \\
9 \\
7 \\
6 \\
\end{bmatrix} \right)\\
&=9
\end{aligned}
$$

A way of thinking of the max function is thinking of it as multiplying every value that is not the max by 0 and multiplying the max by 1 and summing. In this case $\mathbf{y} = 0 \cdot 5 + 1 \cdot 9 + 0 \cdot 7 + 0 \cdot 6$. Which is why we can understand that derivative of $\mathbf{y}$ with respect to $\mathbf{x}$ as such

$$
\frac{\partial y}{\partial \mathbf{x}}=\begin{bmatrix}
0 \\
1 \\
0 \\
0 \\
\end{bmatrix}
$$

This is the same as max-pooling but the difference is that max-pooling has multiple maxes and the function is=of the form $\text{maxpool} :  \mathbb{R}^{N\times M} \rightarrow \mathbb{R}^{K\times L}$. Hence, if we have the matrix $A$, defined as such

$$
\begin{aligned}
A =
\begin{bmatrix}
5 & 2 & 7 & 4\\
6 & 3 & 2 & 1\\
8 & 6 & 5 & 9\\
1 & 3 & 6 & 1\\
\end{bmatrix}
\end{aligned}
$$

If we feed it through maxpool with a 2 by 2 pool, we get a matrix. I will call this $B$ which has the values

$$
\begin{aligned}
B =
\begin{bmatrix}
6 & 7\\
8 & 9\\
\end{bmatrix}
\end{aligned}
$$

We once again use a function as a placeholder $f:\mathbb{R}^{2\times 2} \rightarrow \mathbb{R}$.

$$
\begin{aligned}
B &= \text{maxpool}(A)\\
\frac{\partial c}{\partial A}&=
\begin{bmatrix}
0 & 0 & \frac{\partial c}{\partial b_{12}} & 0\\
\frac{\partial c}{\partial b_{11}} & 0 & 0 & 0\\
\frac{\partial c}{\partial b_{21}} & 0 & 0 & \frac{\partial c}{\partial b_{22}}\\
0 & 0 & 0 & 0\\
\end{bmatrix}
\end{aligned}
$$

Hence, the derivative we get is effectively the partial derivatives of the layers placed in the maximum positions.

## Convolution (Multiple Layer)

One kernel being associated with one layer is quite small, if we had like a 1000 by 1000 image and a 7 by 7 kernel, we would have 49 trainable parameters with 1000000 inputs. While, our 7 by 7 kernel might be able to converge to a reasonable value, a model could benefit with more fittable parameters. We could increase the size of the kernel however this increases problems such as vanishing gradient and it can decrease spatial coherence of the image. What I am getting at is that kernel size is usually increased based on the situation and it is not an inconsequential choice. Hence, the solution is to use multiple kernels for the CNN. This allows the different kernels to learn different patterns in the image allowing for better accuracy of the model while allowing for our input to be multi-channel (such as RGB which is a 3 channel input).

Layers of CNNs can now have an input $n$ channels and the channels are each convolved by the $m$ kernels where $m$ is the number of output channels. This means there are $n\cdot m$ kernels. The convolved channels are then summed point-wise such that each output channel is a sum of every input channel convolved with some kernel.

<img title="" src="file:///C:/Users/zayan/Documents/CNNblogpost/blogposts/images/cnn.png" alt="" width="493" data-align="center">

It can mathematically be understood as such 

$$
O_j = \sum_{i=1}^{n}{I_i*K_{ij}}
$$

where $O_j$ is the $j^\text{th}$ output channel and $I_i$ is the $i^\text{th}$ input channel and $K_{ij}$ is the kernel convolved with $I_i$ which is summed with other convolved channels to form $O_j$

The backpropogation is quite simple as any additional extra terms added to your expression will become 0, when you calculate with respect to ur main term. So the derivates are the same as above, just slightly modified.

$$
\begin{aligned}
\frac{\partial c}{\partial I_i} &= \text{pad}(\frac{\partial c}{\partial O_j}) * \text{flip}(K_{ij})\\
\frac{\partial c}{\partial K_{ij}} &= I_i * \frac{\partial c}{\partial O_j}
\end{aligned}
$$

## Maxpooling (Multiple layer)

Do I really need to elaborate on this, you just do the same maxpooling operation but for every channel, derivatives and everything is all the same.

## Flattening

Flattening is the conversion from multi-channel layer to a vector so that it can be fed into a deep neural network model for a classification task. Values are being moved around as opposed to any addition. Hence, when moving the derivatives backwards, it is just reshaping the derivative vector

<img title="" src="file:///C:/Users/zayan/Documents/CNNblogpost/blogposts/images/flatten.png" alt="" width="378" data-align="center">

## Implementation

## Additional Notes

### 1D Convolution

Convolution is almost always sped up with FFT as mentioned earlier. This means that we can actl

s

s

s

s

s

s

s
afe