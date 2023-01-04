# Building Convolutional Neural Networks from scratch by Karimi Zayan

## Part 2: Convolution and Max Pooling



Hello, welcome to Part 2. If you haven't read part 1, I strongly suggest you read it to gain a deeper understanding about the intuition about matrices more properly. This time, we are ending off the 2 part series with Convolution and Max Pooling Math. By the end of this, I will use all the knowledge over the 2 parts and build a network to do MNIST


##Changing and swapping notation



In the previous article, I stuck to the numerator layout. However, for this article, we may use the denominator layout for convenience as it makes more sense. However, usually, the inbetween on such steps are flat layers and have various ravel operations futher obfuscating the divide between the 2 competing notations.



##Convolution (Single Layer)



Mathematical convolution is defined as



$$(f*g)(t):=\int _{-\infty }^{\infty }f(\tau )g(t-\tau )\,d\tau$$



However, this definition is only for functions which has a domain  $\mathbb{R}$. Essentially, convolution is fancy sliding window multiplication. Convolution works better than a standard deep neural network layer for image because every feature is not analysed pixel wise but a bunch of pixels together are analysed. (local spatial coherence) There is also an order to the inputs, not seen by Deep Neural Network layers. I will not elaborate any further because this article is about the math and not about the rationale. So let us continue. 2D convolution of 2 matrices is defined as such





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

Now let us move onto $\frac{\partial Y}{\partial W}$ by beginning de novo with the individual derivatives. There are 36 of them. While that is a lot, this number will be 112013989021 times large later on so this is like a briefer haha. (pain)

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
&=\text{pad}\left(\frac{\partial c}{\partial {Y}}\right)*\text{rot}(X)

\end{aligned}
$$

Voila, after that pain, we have done it. In this case $\text{pad}$ refers to adding zeroes all around and $\text{rot}$ refers to a 180 degree rotation of the data. The amount of $\text{pad}$ is one less than the size of the kernel.

##Max-pooling (Single Layer)

Now we will find the derivative of a max-pooling layer. Any sort of ravel 



