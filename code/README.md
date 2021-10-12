# Graphene with Hydrogen adatoms :pencil:

## Notes about the script `spin_relaxation.py`

The aim of this script is to provide a benchmark of the Hamiltonian implementation. The idea is to reproduce the results presented by this [paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.081403).  In particular, the results presented in the following figure:

![image-20210921111629177](/home/marcos/.config/Typora/typora-user-images/image-20210921111629177.png)



In this figure we see the spin relaxation time as functions of Fermi level for hydrogenated (upper panels) fluorinated (lower panels) Graphene, both with a concentration of adatoms of $\eta = 53$ ppm of Carbon atoms. In the figure the symbols represent the numerical results from Landauer calculations while the lines come from the T-matrix analytical results.

To calculate the spin relaxation time, we first have to get the spin-flip probability, that is given by:
$$
\Gamma_s(E) = \sum_{\sigma\in\{\pm1\}}\sum_{ij}(|t_{ij;\sigma-\sigma}|^2+|r_{ij;\sigma-\sigma}|^2)
$$
   where *t* and *r* are the transmission and the reflection amplitudes between left-right propagating modes *i* and *j* of opposite spins $\sigma$.

Once $\Gamma_s(E)$ is known, one can get the relaxation time by 
$$
\tau_s^{-1} = \frac{4t}{\hbar}\eta\frac{W}{a}\Gamma_s(E).
$$
  In this expression we have:

* *t* = hopping term between Carbon atoms = 2.6 eV
* $\eta$ = concentration of adatons 
* *W* = width of the strip
* *a* = lattice constant = 2.46 $\AA$

#### Implementation details

**Conservation laws:**

In order to separate the in-coming and out-going states between different spin-projections, one can specify a conservation law for the leads of the system. This is possible because in the leads we don't have spin-orbit terms that couple different spin states, in other words, spin is conserved. The syntax of a conservation law definition is as follows

```python
lead = kwant.Builder(symmetry, conservation_law=matrix)
```

where the `matrix` has to be a matrix that acts on a site and in addition has integer eigenvalues. Internally, Kwant store the blocks in ascending order of the eigenvalues of the conservation law.

After the building phase, the transmission between the conserved blocks are calculated by specifying the both the lead and block  indexes: by instance, if one wants to get the transmission of spin-up into spin-up states, from left lead to right lead, the code will something like:   

```python
 smatrix = kwant.smatrix(system.finalized(), energy, params = syst_params)
 t_up_up = smatrix.transmission((1,i_up), (0,i_up)) # from lead-0 to lead-1
```

 where `i_up` is the index for the spin-up block. If `matrix = -sigma_z`, where `-sigma_z` is given by
$$
-\sigma_z = 
\begin{bmatrix}
-1 & 0\\
0 & 1
\end{bmatrix}
$$
and the Hamiltonian is written in the basis $\{|\uparrow\rangle,|\downarrow\rangle\}$ , the value for `i_up` will 0 (zero), since we're interested in the first block.

#### Implicit summation over the modes

Notice that the we do not say anything about the modes represented by *i* and *j* in the expression for $\Gamma_s(E)$ above. Actually that sum over the indexes *i* and *j* is taken implicitly by kwant. This can be verified by inspection of the source code, but there is also a simple test to probe this. If one desires to get the scattering matrix explicitly, considering all transmission amplitudes of the left-right moving modes, one can use the method `submatrix` of the `smatrix`:

```python
s_from_0_to_1 = smatrix.submatrix(1,0)
```

now `s_from_0_to_1` is a matrix whose entries are the amplitudes $t_{ij;\sigma\sigma^\prime}$. To get the whole transmission, not discriminating the spin-flip, one has to compute
$$
\mathcal{T} = \sum_{ij}\sum_{\sigma\sigma^\prime} |t_{ij;\sigma\sigma^\prime}|^2
$$
   Let's disregard the meaning of the indexes for a moment and call the entries of `s_from_0_to_1` as $\tilde{t}_{ij}$, where the indexes here only represents the indexes for the rows and columns of the matrix that we're going to call by $T$. Following the expression above for the total transmission we can write
$$
\mathcal{T} = ||T||_{F}^2 = \sum^n_i\sum^m_j|\tilde{t}_{ij}|^2
$$
 where we've used the definition for the *Frobenius norm*:
$$
||A||_{F} = \sqrt{\sum^n_i\sum^m_j|\tilde{t}_{ij}|^2}
$$
The result from this detailed calculation will be the same provided by 

```python
T = smatrix.transmission(1,0)
```

To get the matrix and the transmission of specific spin projections, one just has to designate the target and source with tuples:

```python
T_matrix_up_up = smatrix.submatrix((1,0),(0,0)) # target = (1,0), source = (0,0)
T_up_up = np.linalg.norm(T_matrix_up_up)**2 # Squared Frobenius norm
print("spin-up -> spin-up transmission = ", T_up_up)
```

So for the spin-flip we can calculate directly the summation over the modes in a implicitly way.

#### Basis transformation

To study the spin-relaxation time anisotropy, we have to compare results for different spin-projections. Initially, we have to calculate the transmission and reflection amplitudes considering the spin-z flip. With the S-matrix for states written in the spin-z basis it is possible to calculate the S-matrix for other spin-projection basis. 

Consider that we have calculated the S-matrix for the spin-z basis:
$$
\hat{S_z} = 
\begin{bmatrix}
	S_{\uparrow\uparrow} & S_{\uparrow\downarrow}\\
	S_{\downarrow\uparrow} & S_{\downarrow\downarrow}
\end{bmatrix}
$$
Since each of the matrix elements are given by
$$
S_{\sigma^{\prime}\sigma} = \langle\sigma^{\prime}|\hat{S}|\sigma\rangle
$$
with $\sigma = \{\uparrow_z, \downarrow_z\}$, then we can write down the elements for the $\hat{S}_x$ by replacing $|\uparrow(\downarrow)_x\rangle \rightarrow \frac{1}{\sqrt 2} (|\uparrow_z\rangle \pm |\downarrow_z\rangle)$ :
$$
\begin{eqnarray}
	\langle\uparrow_x|S|\uparrow_x\rangle &=& \frac{1}{2} (\langle\uparrow| + \langle \downarrow|) \hat{S} (|\uparrow\rangle + |\downarrow\rangle)\\
	&=& \frac{1}{2} (\langle\uparrow|\hat{S}|\uparrow\rangle + \langle \downarrow|\hat{S}|\uparrow\rangle
    + \langle\uparrow|\hat{S}|\downarrow\rangle + \langle\downarrow|\hat{S}|\downarrow\rangle)
\end{eqnarray}
$$
And, accordingly,  matrix elements for $\hat{S}_y$ are given by adopting $|\uparrow(\downarrow)_y\rangle \rightarrow \frac{1}{\sqrt 2} (|\uparrow_z\rangle \pm i|\downarrow_z\rangle)$: 
$$
\begin{eqnarray}
	\langle\uparrow_y|S|\uparrow_y\rangle &=& \frac{1}{2} (\langle\uparrow| - i\langle \downarrow|) \hat{S} 
	(|\uparrow\rangle +i |\downarrow\rangle)\\
	&=& \frac{1}{2} (\langle\uparrow|\hat{S}|\uparrow\rangle -i \langle \downarrow|\hat{S}|\uparrow\rangle
    +i \langle\uparrow|\hat{S}|\downarrow\rangle + \langle\downarrow|\hat{S}|\downarrow\rangle)
\end{eqnarray}
$$

## Notes on magnetic impurity implementation

### Basis changing and the exchange interaction

The natural next step is to allow the adatoms to have magnetic moment. This is achieved by the inclusion of the exchange interaction given by
$$
H_{\text{ex}} = -J \mathbf{s}\cdot\mathbf{S} = -J(\sigma_x \otimes \Sigma_x + \sigma_y \otimes \Sigma_y +\sigma_z \otimes \Sigma_z)
$$
where $\sigma_i$ are the Pauli matrices acting in the electronic spin-apace while $\Sigma_i$ are Pauli matrices that act in the impurity spin-space. Putting in other words, we have to double the number of degrees of freedom:
$$
\begin{bmatrix}
\uparrow\\
\downarrow
\end{bmatrix} \longrightarrow
\begin{bmatrix}
\uparrow \Uparrow\\
\uparrow \Downarrow\\
\downarrow \Uparrow\\
\downarrow \Downarrow
\end{bmatrix}
$$


In this way, the $H_{\text{ex}}$ can written explicitly as
$$
H_{\text{ex}} = J 
\begin{bmatrix}
	+1&  &  &  \\
      &-1&+2&  \\
      &+2&-1&  \\
      &  &  &+1\\
\end{bmatrix}.
$$
Naturally, both implementation, with and without exchange term, have to return the same results when $J=0$. Our first task is to confirm that it is the case. The best way to certified that everything is working properly is to compare the results of spin relaxation times. Notice that every on-site and hopping term, that were $2\times2$ matrices before, now have to be changed by performing a Kronecker product with an identity matrix:
$$
\sigma_i\longrightarrow \sigma_i \otimes I_{2\times2}
$$
  Moreover, the **conservation law** defined for the leads have to be modified to reflect the enlargement  of the basis and to be possible to have access to every different transmission/reflection amplitude. Let's adopt as our new conservation law the following matrix:
$$
\tilde{\sigma} = 
\begin{bmatrix}
1&&&\\
&2&&\\
&&3&\\
&&&4\\
\end{bmatrix}
$$


### The S-matrix

For the system described above, in which one magnetic impurities are allowed, the scattering matrix will have the following form


$$
\mathbb{S}_{i,j}(E) = 
\begin{bmatrix}
S_{\uparrow\Uparrow,\uparrow\Uparrow} & S_{\uparrow\Uparrow,\uparrow\Downarrow}& S_{\uparrow\Uparrow,\downarrow\Uparrow}& S_{\uparrow\Uparrow,\downarrow\Downarrow}\\
S_{\uparrow\Downarrow,\uparrow\Uparrow} & S_{\uparrow\Downarrow,\uparrow\Downarrow}& S_{\uparrow\Downarrow,\downarrow\Uparrow}& S_{\uparrow\Downarrow,\downarrow\Downarrow}\\
S_{\downarrow\Uparrow,\uparrow\Uparrow} & S_{\downarrow\Uparrow,\uparrow\Downarrow}& S_{\downarrow\Uparrow,\downarrow\Uparrow}& S_{\downarrow\Uparrow,\downarrow\Downarrow}\\
S_{\downarrow\Downarrow,\uparrow\Uparrow} & S_{\downarrow\Downarrow,\uparrow\Downarrow}& S_{\downarrow\Downarrow,\downarrow\Uparrow}& S_{\downarrow\Downarrow,\downarrow\Downarrow}
\end{bmatrix}
$$
  where $i(j)$ identify the target (source) lead, $E$ is the energy or chemical potential. As we've seen before, when $i=j$ the matrix elements will represent the reflection amplitudes ($S\rightarrow R$) while the case where $i\neq j$ will give us the transmission amplitudes ($S\rightarrow T$).

We are primarily interested in the electronic spin-flip transmissions and reflections, i. e.
$$
T_{\uparrow,\downarrow},~T_{\downarrow,\uparrow},~R_{\uparrow,\downarrow},~\text{and }R_{\downarrow,\uparrow}.
$$
To get rid of the impurity spin degree of freedom, **one have to trace it out**. For now it  may appear abstract, but it will be clear after few lines from bellow. Let's see how we can get the amplitudes above from [Kwant](https://kwant-project.org/) results:

* By the choice of the conservation law, and basis, we have the following mapping for the states:
  $$
  \{0,1,2,3\} \longrightarrow \{ \Ket{\uparrow\Uparrow}, \Ket{\uparrow\Downarrow}, \Ket{\downarrow\Uparrow}, \Ket{\downarrow\Downarrow} \}
  $$

* The transmission (reflection) amplitudes are given by the partial trace over the impurity spin space, considering the density matrix of an unpolarized state:
  $$
  \rho_B = \frac{1}{2}(\Ket{\Uparrow}\Bra{\Uparrow} + \Ket{\Downarrow}\Bra{{\Downarrow}})
  $$

* The partial trace over the impurity spin space is performed as follows:
  $$
  \begin{eqnarray}
  \hat{S}'_A &=& \text{tr}_B[\hat S \hat \rho_B] \\ 
  &=& \text{tr}_B[(\hat S_A \otimes \hat S_B)(1_A\otimes\hat \rho_B)]\\
  &=& \hat S_A ~\text{tr}[\hat S_B \hat \rho_B]
  \end{eqnarray}
  $$

* More explicitly, we have
  $$
  \begin{eqnarray}
  \text{tr}(\hat S_B \hat \rho_B) &=& \sum_{i= \{ \Uparrow, \Downarrow\}} \Bra{i}\hat S_B \hat \rho_B\Ket{i}\\
  &=& \sum_{i,j = \{ \Uparrow, \Downarrow\}} \Bra{i}\hat S_B \Ket{j}\Bra{j} \hat \rho_B\Ket{i}\\
  &=& \frac{1}{2}\Bra{\Uparrow}\hat S_B \Ket{\Uparrow} + \frac{1}{2}\Bra{\Downarrow}\hat S_B \Ket{\Downarrow} 
  \end{eqnarray}
  $$
  
  $$
  \begin{eqnarray}
  \hat S'_A &=& \frac{1}{2}
  \begin{bmatrix}
  \bra{\uparrow}S_A\ket{\uparrow}&\bra{\uparrow}S_A\ket{\downarrow}\\
  \bra{\downarrow}S_A\ket{\uparrow}&\bra{\downarrow}S_A\ket{\downarrow}\\
  \end{bmatrix} \cdot [\Bra{\Uparrow}\hat S_B \Ket{\Uparrow} + \Bra{\Downarrow}\hat S_B \Ket{\Downarrow}] \\
  &=& \frac{1}{2}
  \begin{bmatrix}
  \bra{\uparrow\Uparrow}(S_A\otimes S_B)\ket{\uparrow\Uparrow} + \bra{\uparrow\Downarrow}(S_A\otimes S_B)\ket{\uparrow\Downarrow} 
  & 
  \bra{\uparrow\Uparrow}(S_A\otimes S_B)\ket{\downarrow\Uparrow} + \bra{\uparrow\Downarrow}(S_A\otimes S_B)\ket{\downarrow\Downarrow}
  \\
  \bra{\downarrow\Uparrow}(S_A\otimes S_B)\ket{\uparrow\Uparrow} + \bra{\downarrow\Downarrow}(S_A\otimes S_B)\ket{\uparrow\Downarrow}
  &
  \bra{\downarrow\Uparrow}(S_A\otimes S_B)\ket{\uparrow\Uparrow} + \bra{\downarrow\Downarrow}(S_A\otimes S_B)\ket{\downarrow\Downarrow}\\
  \end{bmatrix} \\
  &=& \frac{1}{2}
  \begin{bmatrix}
  S_{\uparrow\Uparrow, \uparrow\Uparrow} + S_{\uparrow\Downarrow, \uparrow\Downarrow} 
  & 
  S_{\uparrow\Uparrow, \downarrow\Uparrow} + S_{\uparrow\Downarrow,\downarrow\Downarrow}
  \\
  S_{\downarrow\Uparrow, \uparrow\Uparrow} + S_{\downarrow\Downarrow,\uparrow\Downarrow}
  &
  S_{\downarrow\Uparrow, \downarrow\Uparrow} + S_{\downarrow\Downarrow, \downarrow\Downarrow}\\
  \end{bmatrix}
  \end{eqnarray}
  $$




Translating into the Kwant syntax, where we have: 

* $S_{\uparrow\Uparrow, \uparrow\Uparrow} = $ `smatrix.submatrix((i,0), (j,0))`, $S_{\uparrow\Downarrow, \uparrow\Downarrow} = $ `smatrix.submatrix((i,1), (j,1))`

* $S_{\uparrow\Uparrow, \downarrow\Uparrow} = $ `smatrix.submatrix((i,0), (j,2))`, $S_{\uparrow\Downarrow, \downarrow\Downarrow} = $ `smatrix.submatrix((i,1), (j,3))`

* $S_{\downarrow\Uparrow, \uparrow\Uparrow} = $ `smatrix.submatrix((i,2), (j,0))`, $S_{\downarrow\Downarrow, \uparrow\Downarrow} = $ `smatrix.submatrix((i,3), (j,1))`

* $S_{\downarrow \Uparrow, \uparrow\Uparrow} = $ `smatrix.submatrix((i,2), (j,2))`, $S_{\downarrow\Downarrow, \downarrow\Downarrow} = $ `smatrix.submatrix((i,3), (j,3))`


The matrix elements that will be important to us (those with electron spin flip)


```python
smatrix = kwant.smatrix(system.finalized(), energy, params=parameters_hamiltonian)

T_up_dn = 1/2 * (smatrix.submatrix((1,0), (0,2)) + smatrix.submatrix((1,1), (0,3)))
T_dn_up = 1/2 * (smatrix.submatrix((1,2), (0,0)) + smatrix.submatrix((1,3), (0,1)))

R_up_dn = 1/2 * (smatrix.submatrix((0,0), (0,2)) + smatrix.submatrix((0,1), (0,3)))
R_dn_up = 1/2 * (smatrix.submatrix((0,2), (0,0)) + smatrix.submatrix((0,3), (0,1)))
```

 

####  Angular momentum conservation

With the procedure depicted above, we could describe well the situation when the exchange interaction is not considered. However, when we include the magnetic moment due to adatom's spin, we have to think about the angular momentum conservation. In this situation the electron spin-flip is accompanied by the adatom spin-flip as well. The scattering amplitudes that describe this scenario are 

* $S_{\uparrow\Downarrow,\downarrow\Uparrow}=$`smatrix.submatrix((i,1),(j,2))`

* $S_{\downarrow\Uparrow,\uparrow\Downarrow}=$`smatrix.submatrix((i,2),(j,1))`

 Considering an unpolarized system, we will get the spin-flip amplitude as the average of both process above:
$$
S_{\text{flip}} = \frac{1}{2} \left\{ S_{\uparrow\Downarrow,\downarrow\Uparrow} + S_{\downarrow\Uparrow,\uparrow\Downarrow} \right\}
$$
