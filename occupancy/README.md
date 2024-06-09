# 2d to 3d

Checkout `morph2d.py`. 

```
python morph2d.py --imgSize 100 
```

# 3d to 4d

Checkout `morph3d.py`

```
python morph3d.py --imgSize 100 --shape1 0 --shape2 2
```

- shape 0 is a cube.
- shape 1 is a ball.
- shape 2 is a cylinder.
- shape 3 is a tetrahedron.

# 3d to 5d

Checkout `morph3dto5d.py`

```
python morph3dto5d.py --imgSize 100 --shape1 0 --shape2 2 --divide 2
```

- shape 0 is a cube.
- shape 1 is a ball.
- shape 2 is a cylinder.
- shape 3 is a tetrahedron.
- `--divide` is used to reduce the memory pressure.
