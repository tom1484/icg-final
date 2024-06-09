# 2024 ICG Final Project - G61

# Install Dependencies

```bash
pip install -r requirements.txt
```

# Run Tests

## Mesh approach

```bash
# Generate test 4D mesh
python -m mesh.tests.generate_4D ./mesh/result/sculpt_4D.obj
# Render 3D projection of the 4D mesh to GIF
python -m mesh.tests.render_4D ./mesh/result/sculpt_4D.obj ./mesh/result/sculpt_4D.gif
```

## Occupancy approach

```bash
```

# Division of work:

- **Mesh approach**:
  - **Mesh generation**: Chu-Rong Chen
  - **Mesh intersections**: Chu-Rong Chen
  - **Mesh cutting**: Chu-Rong Chen
- **Occupancy approach**:
