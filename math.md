
http://sysplan.nams.kyushu-u.ac.jp/gen/papers/paper2012/A_BasisOfRL.pdf
http://sigfin.org/?plugin=attach&refer=SIG-FIN-003-11&openfile=SIG-FIN-003-11.pdf
<br>

$$
\Large
\begin{eqnarray}

Q^\pi(s, a) &=& \mathbb{ E }_{\pi,Pr}[\sum_{ t = 0 }^{ \infty } \gamma^t R(s_t, a_t, s_{t+1})|s_0=s, a_0=a] \\

&=& \mathbb{ E }_{\pi,Pr}[R(s_0, a_0, s_1) + \sum_{ t = 1 }^{ \infty } \gamma^t R(s_t, a_t, s_{t+1})|s_0=s, a_0=a] \\

&=& \mathbb{ E }_{\pi,Pr}[R(s_0, a_0, s_1)|s_0=s, a_0=a] \\
&+& \mathbb{ E }_{\pi,Pr}[\sum_{ t = 1 }^{ \infty } \gamma^t R(s_t, a_t, s_{t+1})|s_0=s, a_0=a] \\
&=& \mathbb{ E }_{Pr(s'|s,a)}\mathbb{ E }_{\pi(a'|s')}[R(s, a, s') + \gamma Q^\pi(s', a')]
\end{eqnarray}
$$

<br>
