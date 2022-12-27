# µΩ - Just a little bit of resistance ...

A prototype of a SI unit system implementation using the new [Numpy extensible data type](https://numpy.org/neps/nep-0042-new-dtypes.html) feature that aims to solve the cross-library incompatibility problem that other unit system implementations, like [unyt](https://github.com/yt-project/unyt), suffer.

At a high level, the implementation consists of a `Quantity` scalar object and a `QuantityDType` Numpy DTypeMeta. The scalar object contains all of the unit system logic and scalar arithmetic. The dtype object allows the native Numpy `ndarray` to represent floating point values in the array as `Quantity` objects.

`Quantity` converts the input quantity to the equivalent base SI units. This is how Mathcad works.

```python
>>> from microohm import Quantity
>>> v = Quantity(1, "mV")
>>> v
Quantity('0.001 V')
>>> v.display(".1f mV")
1.0 mV
>>> i = Quantity("1 kA")
>>> i
Quantity('1000.0 A')
>>> r = v/i
>>> r
Quantity('1e-06 Ω')
>>> r.display(".1f µΩ")
1.0 µΩ
```

Use `QuantityDType` to instantiate an array of floating point values with a given unit.

```python
>>> from microohm import QuantityDType
>>> import numpy as np
>>> v_arr = np.array([1.0, 2.0], dtype=QuantityDType("mV"))
>>> i_arr = np.array([1.0, 2.0], dtype=QuantityDType("kA"))
>>> r_arr = v_arr/i_arr
>>> r_arr
array([Quantity('1e-06 Ω'), Quantity('1e-06 Ω')],
      dtype=QuantityDType('mV/(kA)'))
>>> r_arr[0].display(".1f µΩ")
1.0 µΩ
```

There is support for non-SI system units as defined in [NIST Special Publication 811](https://www.nist.gov/pml/special-publication-811).

```python
>>> d1 = Quantity(1, "mil")
>>> d2 = Quantity(1, "mm")
>>> d3 = d1 + d2
>>> d3
Quantity('0.0010254 m')
>>> d3.display(".6f mil")
40.370079 mil
```

This approach solves the cross-library incompatibility problem.

```python
>>> import pandas as pd
>>> arr = np.array([1, 2], dtype=QuantityDType("mV/Hz**(1/2)"))
>>> s1 = pd.Series(arr)
>>> s1.iloc[0].display(".1f mV/Hz**(1/2)")
1.0 mV/Hz**(1/2)
>>> s2 = 2 * s1
>>> s2.iloc[1].display(".1f mV/Hz**(1/2)")
2.0 mV/Hz**(1/2)
```

## Developer workflow
The `QuantityDType` has to be implemented using the Numpy C-API and compiled as a C-extension module. The following flow works on Ubuntu 22.04.

```bash
$ git clone git@github.com:l-johnston/microohm.git
$ cd microohm
$ python -m venv .venv
$ source .venv/bin/activate
(.venv) $ python -m pip install -U pip
(.venv) $ pip install -U setuptools
(.venv) $ pip install -e .[dev]
```
