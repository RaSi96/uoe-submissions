# Fokker-Planck Derivation

**Main source: https://www.youtube.com/watch?v=MmcgT6-lBoY**

**Supplementary: https://www.thphys.uni-heidelberg.de/~wolschin/statsem21_6s.pdf**

## Brownian Motion
We'll use a combination of Ito's Lemma and Expectations. The Einstein diffusion equation is a Fokker-Planck (F_P) solution to Brownian motion. In general with F-P, we're interested in how our distribution evolves over time, not so much deterministic evolution. Start with simple Brownian motion:
$$ dB_t = dW_t $$

Recall that $W_t$ is the continuous Wiener process, which is the z-standardised (scaled) limit at infinite steps ($N \to \infty$) of a Random Walk (RW). At infinite steps, the distribution of RW increments goes from a per-step Bernoulli over $\{+1, -1\}$ to an i.i.d Gaussian with $\mu=0, \text{Var}=N$. If we let $N \to \infty$, we end up taking infinite(simally small) steps from one time $t$ to the next $t+1$; thus, $dW_t$ is now normally distributed with $\mu=0, \text{Var} = dt$. Standard deviation of $dW_t = \sqrt{\text{Var}} = \sqrt{dt}$. As such:
$$ B_{t+1} - B_t \sim N(0, \Delta t) $$

For notational convenience we'll define
$$ X_t := B_t $$

Making our differential equation:
$$ dX_t = dB_t $$

First let's say we have an arbitrary, smooth function in X, $f(X_t)$, such that it's continuous, twice-differentiable, and has compact support. Compact support means that ther exists some $x_1, x_2$ values, where $x_1 < x < x_2$, such that $f(x)=0$ outside of this interval. This also means $f'(x) = f''(x) = 0$ for $x<x_1, x>x_2$ as well. An example of this is an options portfolio: outside of some market range, profit is 0 (the portfolio is loss-making); within a certain market range, profit is $x$.

Next, we apply Ito's lemma to $f(X_t)$:
$$ df = f_x dX_t + \frac{1}{2} f_{xx} dX_t^2 $$

$dX_t^2$ is easy to calculate, it's just $dt$ (a property of the Wiener process). Now we plug $dX_t=dB_t, dX_t^2=dt$ back into Ito's lemma to get:
$$ df = f_x dB_t + \frac{1}{2} f_{xx} dt $$

A few points to note here:
1. We're interested in the evolution of the SDE's distribution over time, not so much anything else. As such, in this case we want to get rid of $dB_t$ - the stochastic forcing. We remove stochastic terms by taking an expectation, $\mathbb{E}[X]$. A pertinent question is, how? We're collapsing an entire distribution into a single number, how do we go back to a distribution from it? If integrals and derivatives are inverses, we must have an inverse for this as well - and indeed we do, because recall an expectation operator isn't _just_ the fancy $\mathbb{E}$, it's an integral of the function multiplied with its probabilty density:
   $$ \int_{-\infty}^{+\infty} f(x) p(x) dx $$

   Because of this, we can easily manipulate our equations back and forth to isolate what we need, use different PDFs aside from just a Gaussian, use quantiles or medians instead of averages, etc.
2. One might argue that by getting rid of $dB_t$ we're losing a source of stochasticity, but one must also understand that we _started_ with pure $dB_t$ and applied Ito's Lemma, which is a stochastic Taylor expansion. That's where we got two terms from; getting rid of $dB_t$ doesn't mean we're losing information about our dsitributon, that information is already captured in the second $1/2$ fractional 2nd derivative. The first $dB_t$ is effectively useless.

So, we remove the stochastic term by applying the expectation operator to both sides; because $dB_t$ has $\mu=0$, it disappears:

\begin{align*}
    \mathbb{E}[df] &= \mathbb{E} \left[ f_x dB_t + \frac{1}{2} f_{xx} dt \right]\\
                   &= \frac{1}{2} \mathbb{E}[f_{xx}] dt
\end{align*}

Now we move $dt$ to the other side and rearrange:
$$ \frac{d}{dt} \mathbb{E}[f] = \frac{1}{2} \mathbb{E}[f_{xx}] $$

That move and rearranging can be justified via the Dominated Convergence Theorem. Now we rewrite our expectation as its integral, involving the PDF of $x$. Let $p(x, t)$ denote the PDF of $x$ at time $t$, meaning we can now:
$$ \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x,t) dx = \frac{1}{2} \int_{-\infty}^{+\infty} f_{xx}(x) p(x, t) dx $$

We need to simplify the right-hand side. We have the 2nd derivative of $f$ with respect to $x$, and we want to get rid of the derivative of $f$. We can use integration by parts:
$$ \int u dv = uv - \int v du $$

We'll let $dv = f_{xx}(x) dx$ and $u = p(x, t)$. This gives us $v = f_x(x) dx$ and $du = \frac{\partial p(x, t)}{\partial x} dx $. Now we can plug this into our previous expression:
$$ \int_{-\infty}^{+\infty} f_{xx}(x) p(x, t) dx = f_x(x) p(x, t) \vert_{x=-\infty}^{x=+\infty} - \int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx $$

We can easily discard $f(x)$ later on because it's an arbitrary function; F-P only depends on $p(x, t)$'s evolution. Now, we know that probabilities tail off really quickly, meaning at $\pm \infty$, $f_x(x) p(x, t) \vert_{x=-\infty}^{x=+\infty} = 0$. We rewrite:
$$ \int_{-\infty}^{+\infty} f_{xx}(x) p(x, t) dx = -\int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx $$

\begin{align*}
    \int_{-\infty}^{+\infty} f_{xx}(x) p(x, t) dx &= -\int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx \\
    \implies \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x,t) dx &= - \frac{1}{2} \int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx
\end{align*}

We need to apply integration by parts again to get $f(x)$ onto the right hand side. Let $dv := f_x(x) dx, \; u := \frac{\partial p(x, t)}{\partial x}$, so we get $v = f(x), \; du = \frac{\partial^2 p(x, t)}{\partial x^2} dx$. Now we substitute:
$$ \int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx = f(x) \frac{\partial p(x, t)}{\partial x} \vert_{x=-\infty}^{x=+\infty} - \int_{-\infty}^{+\infty} f(x) \frac{\partial^2 p(x, t)}{\partial x^2} dx $$

The first term is again $0$ since probabilities tail off, so:

\begin{align*}
    \int_{-\infty}^{+\infty} f_x(x) \frac{\partial p(x, t)}{\partial x} dx &= - \int_{-\infty}^{+\infty} f(x) \frac{\partial^2 p(x, t)}{\partial x^2} dx \\
    \implies \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x,t) dx &= \frac{1}{2} \int_{-\infty}^{+\infty} f(x) \frac{\partial^2 p(x, t)}{\partial x^2} dx
\end{align*}

Now we rearrange and combine integrands:

\begin{align*}
    \int_{-\infty}^{+\infty} f(x) \frac{\partial p(x,t)}{\partial t} dx &= \frac{1}{2} \int_{-\infty}^{+\infty} f(x) \frac{\partial^2 p(x, t)}{\partial x^2} dx \\
    \implies \int_{-\infty}^{+\infty} f(x) \left( \frac{\partial p(x,t)}{\partial t} - \frac{1}{2} \frac{\partial^2 p(x, t)}{\partial x^2} \right)  dx &= 0
\end{align*}

Now since $f(x)$ is arbitrary, if the integral must equal 0 then the terms within the brackets must be zero. This gives us the F-P equation for standard Brownian motion, or the diffusion equation:
$$ \boxed{ \therefore \frac{\partial p(x,t)}{\partial t} - \frac{1}{2} \frac{\partial^2 p(x, t)}{\partial x^2} = 0 } $$

## General SDE
Great. Now what about the general case? We know SDEs that are more complicated than Brownian motion. Let's look at one with non-constant drift & diffusion:
$$ dX_t = \mu(X_t, t) dt + \sigma (X_t, t) dt $$

Just as before, define an arbitrary smooth, continuous, twice-differentiable, compact function $f(X_t)$ and apply Ito's lemma to start with:

\begin{align*}
    df &= f_x dX_t + \frac{1}{2} f_xx dX_t^2 \\
       &= f_x (\mu dt + \sigma dB_t) + \frac{1}{2} \sigma^2 f_{xx} dt
\end{align*}

We'll drop an explicit dependence on time $t$ here and reintroduce it later for notational convenience. This is also where we bring expectations in to help remove stochastic terms:

\begin{align*}
       &= \left( \mu f_x + \frac{1}{2} \sigma^2 f_{xx} \right) dt + f_x \sigma dB_t \\
       \\
    \mathbb{E}[df] &= \mathbb{E} \left( \mu f_x + \frac{1}{2} \sigma^2 f_{xx} \right) dt + f_x \sigma dB_t\\
                   &- \mathbb{E} \left[ \mu f_x + \frac{1}{2} \sigma^2 f_{xx} \right] dt\\
        \\
    \frac{d}{dt} \mathbb{E}[f] &= \mathbb{E} \left[ \mu f_x + \frac{1}{2} \sigma^2 f_{xx} \right]
\end{align*}

Now we reintroduce $p(x, t)$ as the probability of $x$ at time $t$ (remember, $p(x, t)$ is the main object we're after). We'll now expand expectations into PDFs, and also reintroduce $t$ in the notation for $\mu, \sigma$:

\begin{align*}
    \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x, t) dx &= \int_{-\infty}^{+\infty} \left( \mu(x, t) f_x(x) + \frac{1}{2} \sigma^2(x, t) f_{xx}(x) \right) p(x, t) dx\\
    &=  \int_{-\infty}^{+\infty} \mu(x, t) f_x(x) p(x, t) dx + \frac{1}{2} \int_{-\infty}^{+\infty} \sigma^2(x, t) f_{xx}(x) p(x, t) dx
\end{align*}

We need to apply integration by parts to both integrals on the RHS now, separately. Let's start with the first term involving $\mu$:

\begin{align*}
    dv := f_x(x) dx; &\quad  u := \mu(x, t)p(x, t)\\
     v  = f(x)     ; &\quad du  = \frac{\partial}{\partial dx} (\mu(x,t) p(x,t)) dx
\end{align*}

And now back-substitute, remembering that the first term at limits $\pm \infty$ will be zero since probabilities are zero at $\infty$:

\begin{align*}
    \implies \int_{-\infty}^{+\infty} \mu(x, t) f_x(x) p(x, t) dx &= f(x) \mu(x, t) p(x, t) \vert_{x=-\infty}^{x=+\infty} - \int_{-\infty}^{+\infty} f(x) \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) dx \\
    &= - \int_{-\infty}^{+\infty} f(x) \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) dx
\end{align*}

Now for the second term involving $\sigma$. We have a 2nd derivative in there so we need integration by parts twice:

\begin{align*}
    dv := f_{xx}(x) dx; &\quad  u := \sigma^2(x, t)p(x, t)\\
     v  = f_x(x)      ; &\quad du  = \frac{\partial}{\partial dx} (\sigma^2(x,t) p(x,t)) dx
\end{align*}

The first term is zero again:

\begin{align*}
    \implies \frac{1}{2} \int_{-\infty}^{+\infty} \sigma^2(x, t) f_{xx}(x) p(x, t) dx &= f_x(x) \sigma^2(x, t) p(x, t) \vert_{x=-\infty}^{x=+\infty} - \int_{-\infty}^{+\infty} f_x(x) \frac{\partial}{\partial x} (\sigma^2(x, t) p(x, t)) dx \\
    &= - \int_{-\infty}^{+\infty} f_x(x) \frac{\partial}{\partial x} (\sigma^2(x, t) p(x, t)) dx
\end{align*}

We do integration by parts again - basically reduce the derivative of $f(x)$ and increase the partial by 1:

$$ - \int_{-\infty}^{+\infty} f_x(x) \frac{\partial}{\partial x} (\sigma^2(x, t) p(x, t)) dx = \frac{1}{2} \int_{-\infty}^{+\infty} f(x) \frac{\partial^2}{\partial x^2} (\sigma^2(x, t) p(x, t)) dx $$

And now we rewrite the big equation:
$$ \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x, t) dx = - \int_{-\infty}^{+\infty} f(x) \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) dx + \frac{1}{2} \int_{-\infty}^{+\infty} f(x) \frac{\partial^2}{\partial x^2} (\sigma^2(x, t) p(x, t)) dx $$

On the LHS, we can rearrange the derivative to get:

\begin{align*}
    \frac{d}{dt} \int_{-\infty}^{+\infty} f(x) p(x, t) dx &= \int_{-\infty}^{+\infty} f(x) \frac{\partial p(x, t)}{\partial t} dx \\
    \implies \int_{-\infty}^{+\infty} f(x) \frac{\partial p(x, t)}{\partial t} dx &= - \int_{-\infty}^{+\infty} f(x) \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) dx + \frac{1}{2} \int_{-\infty}^{+\infty} f(x) \frac{\partial^2}{\partial x^2} (\sigma^2(x, t) p(x, t)) dx
\end{align*}

And now we combine all our integrals on one side:
$$ \int_{-\infty}^{+\infty} f(x)\ left( \frac{\partial p(x, t)}{\partial t} + \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) - \frac{1}{2} \frac{\partial^2}{\partial x^2} (\sigma^2(x, t) p(x, t)) \right) dx = 0 $$

And just like last time, since $f(x)$ is arbitrary, if the integral must equal 0 then the stuff in brackets must be zero. This gives us the general Fokker-Planck equation:
$$ \boxed{ \therefore \frac{\partial p(x, t)}{\partial t} + \frac{\partial}{\partial x} (\mu(x, t) p(x, t)) - \frac{1}{2} \frac{\partial^2}{\partial x^2} (\sigma^2(x, t) p(x, t)) = 0 } $$

The general derivation actually lets us inspect Fokker-Planck solutions to a variety of SDEs, including but not limited to O-U, GBM, B-S, etc. We just need to substitute the correct $\mu$ and $\sigma$ terms from the SDE into the general solution.

## Conclusion

As mentioned, in any case, just pick an arbitrary smooth, continuous, twice-differentiable, compact function $f(X_t)$ and apply Ito's Lemma. Continue from there, and don't forget to integrate by parts!

There is a _whole lot more_ to the Fokker-Planck equation than what we have here, including specific terminology (semi-group, infinitesimal generators (which are very important), Kramers-Moyal expansion, etc.) All of this is quite a bit to include here right now, so I'd suggest checking out the source material for all this.

# References