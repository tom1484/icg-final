# 2024 ICG Final Project - G61

## Authors
- [Chu-Rong Chen](https://github.com/tom1484)
- [Chun-Mao Lai](https://github.com/Mecoli1219)

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Tests

### Mesh approach

```bash
# Generate test 4D mesh
python -m mesh.tests.generate_4D ./mesh/result/sculpt_4D.obj
# Render 3D projection of the 4D mesh to GIF
python -m mesh.tests.render_4D ./mesh/result/sculpt_4D.obj ./mesh/result/sculpt_4D.gif
```

### Occupancy approach

Refer to `occupancy/README.md` for more details.

## Division of work

- **Mesh approach**: Chu-Rong Chen (tom1484)
- **Occupancy approach**: Chun-Mao Lai (Mecoli1219)
