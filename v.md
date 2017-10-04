http://yamaimo.hatenablog.jp/entry/2015/09/03/200000

https://www.google.co.jp/search?q=Reinforcement+Learning%E3%80%80value+function+formula+&btnG=%E6%A4%9C%E7%B4%A2&client=firefox-b&dcr=0&gbv=1

https://www.google.co.jp/search?q=Reinforcement+Learning%E3%80%80value+function+derivation+of+an+equation&btnG=%E6%A4%9C%E7%B4%A2&client=firefox-b&dcr=0&gbv=1

https://stats.stackexchange.com/questions/243384/deriving-bellmans-equation-in-reinforcement-learning

https://www.reddit.com/r/reinforcementlearning/comments/4puel5/derivation_of_the_bellman_equation_for_values/

$$
\begin{align}
\mathcal{P}_{ss'}^{a} &= Pr\{s_{t+1} = s' | s_t = s, a_t = a \} \\
\mathcal{R}^{a}_{ss'} &= E\{r_{t+1} | s_t = s, a_t = a, s_{t+1} = s' \}
\end{align}
$$

<br>

$$
\begin{align}
E \{ r_{t+1} | s_t = s \}
&=\sum_{a \in \mathcal{A}(s)} \sum_{s' \in \mathcal{S}}(E\{r_{t+1} | s_t = s, a_t = a, s_{t+1} = s' \} ) \\
&=\sum_{a \in \mathcal{A}(s)} \sum_{s' \in \mathcal{S}}\mathcal{R}^{a}_{ss'}
\end{align}
$$
$$
\begin{align}
E \{\gamma ( r_{t+2} + \gamma r_{t+3} + \cdots ) | s_t = s \} \\
&= \gamma E \{r_{t+2} + \gamma r_{t+3} + \cdots | s_t = s \} \\
&= \gamma E \{R_{t+1} | s_t = s \} \\

&= \gamma E \{E \{R_{t+1}| s_t=s'\} | s_t = s \} \\
&= \gamma E \{E \{R_{t+1}| s_t=s'\} | s_t = s \} \\

&= \gamma \sum_{a \in \mathcal{A}(s)} \sum_{s' \in \mathcal{S}}(E \{R_{t+1} | s_{t+1} = s',  a_t = a, s_t = s \} )
\end{align}
$$


<br>

$$
\begin{align}
V^{\pi}_s &= E \left\{ \left. R_t \right| s_t = s \right\} \\
&= E \left\{ \left. r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots \right| s_t = s \right\} \\
&= E \left\{ \left. r_{t+1} + \gamma \left( r_{t+2} + \gamma r_{t+3} + \cdots \right) \right| s_t = s \right\} \\

&= \sum_{a \in \mathcal{A}(s)} \pi(s, a) \left( \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} \left( \mathcal{R}^a_{ss'} + \gamma E \left\{ \left. R_{t+1} \right| s_{t+1} = s' \right\} \right) \right) \\
&= \sum_{a \in \mathcal{A}(s)} \pi(s, a) \left( \sum_{s' \in \mathcal{S}} \mathcal{P}^a_{ss'} \left( \mathcal{R}^a_{ss'} + \gamma V^{\pi}_{s'} \right) \right)
\end{align}
$$
