import numpy as np
import pytest
import mritk.data.orientation


def test_apply_affine_identity():
    """Test that applying an identity matrix returns the original points."""
    points = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    identity_affine = np.eye(4)

    result = mritk.data.orientation.apply_affine(identity_affine, points)

    np.testing.assert_array_equal(result, points)


def test_apply_affine_translation():
    """Test translation logic: x' = x + t."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    affine = np.eye(4)
    # Set translation vector [Tx, Ty, Tz]
    translation = np.array([5.0, 10.0, -5.0])
    affine[:3, 3] = translation

    expected = points + translation
    result = mritk.data.orientation.apply_affine(affine, points)

    np.testing.assert_array_almost_equal(result, expected)


def test_apply_affine_scaling():
    """Test scaling logic: x' = s * x."""
    points = np.array([[1.0, 2.0, 3.0]])
    # Scale x*2, y*0.5, z*-1
    affine = np.diag([2.0, 0.5, -1.0, 1.0])

    expected = np.array([[2.0, 1.0, -3.0]])
    result = mritk.data.orientation.apply_affine(affine, points)

    np.testing.assert_array_almost_equal(result, expected)


def test_physical_to_voxel_indices_basic_translation():
    """
    Test transforming world coordinates back to voxel coordinates.
    Scenario: The affine translates voxel space by +10.
    Therefore, a world coordinate of 10 should map back to voxel 0.
    """
    # World coordinates (DOFs)
    dof_coords = np.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])

    # Affine that adds 10 to everything
    affine = np.eye(4)
    affine[:3, 3] = [10.0, 10.0, 10.0]

    # We expect the function to apply the INVERSE of (+10) -> (-10)
    # 10 - 10 = 0
    # 11 - 10 = 1
    expected = np.array([[0, 0, 0], [1, 1, 1]])

    result = mritk.data.orientation.physical_to_voxel_indices(dof_coords, affine, round_coords=True)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == int


def test_physical_to_voxel_indices_no_rounding():
    """Test that floating point results are returned when rint=False."""
    dof_coords = np.array([[10.5, 10.5, 10.5]])
    affine = np.eye(4)  # Identity

    result = mritk.data.orientation.physical_to_voxel_indices(dof_coords, affine, round_coords=False)

    np.testing.assert_array_almost_equal(result, dof_coords)
    assert np.issubdtype(result.dtype, np.floating)


def test_physical_to_voxel_indices_rounding_behavior():
    """Test that rint rounds correctly."""
    # 10.1 -> 10, 10.9 -> 11
    dof_coords = np.array([[10.1, 10.1, 10.1], [10.9, 10.9, 10.9]])
    affine = np.eye(4)

    expected = np.array([[10, 10, 10], [11, 11, 11]])

    result = mritk.data.orientation.physical_to_voxel_indices(dof_coords, affine, round_coords=True)
    np.testing.assert_array_equal(result, expected)


def test_find_nearest_valid_voxels_1_neighbor():
    """Test finding the single closest point in a 2D mask."""
    # Define a mask with valid pixels only at (0,0) and (5,5)
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True
    mask[5, 5] = True

    # Point A is close to (0,0), Point B is close to (5,5)
    dof_inds = np.array([[0.1, 0.1], [4.9, 4.9]])

    # Function output shape is (ndim, N_neighbors, N_points)
    result = mritk.data.orientation.find_nearest_valid_voxels(dof_inds, mask, k=1)

    # Verify shape: (2 dims, 1 neighbor, 2 query points)
    assert result.shape == (2, 1, 2)

    # First point (0.1, 0.1) -> Neighbor should be (0, 0)
    np.testing.assert_array_equal(result[:, 0, 0], [0, 0])
    # Second point (4.9, 4.9) -> Neighbor should be (5, 5)
    np.testing.assert_array_equal(result[:, 0, 1], [5, 5])


def test_find_nearest_valid_voxels_N_neighbors():
    """Test finding multiple neighbors (N=2) in 3D."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    # Two valid points close to each other
    mask[1, 1, 1] = True
    mask[1, 1, 2] = True
    # One valid point far away
    mask[9, 9, 9] = True

    # Query point right next to the cluster at (1,1,1)
    dof_inds = np.array([[1.0, 1.0, 1.1]])

    result = mritk.data.orientation.find_nearest_valid_voxels(dof_inds, mask, k=2)

    # Shape should be (3 dims, 2 neighbors, 1 point)
    assert result.shape == (3, 2, 1)

    # Get the neighbors for the first (and only) query point
    neighbors = result[:, :, 0].T  # Transpose to get list of coords: shape (2, 3)

    # We expect (1,1,1) and (1,1,2) to be the neighbors.
    # KDTree returns sorted by distance.
    # Distance to (1,1,1) is 0.1
    # Distance to (1,1,2) is 0.9
    # So (1,1,1) should be first.
    np.testing.assert_array_equal(neighbors[0], [1, 1, 1])
    np.testing.assert_array_equal(neighbors[1], [1, 1, 2])


def test_find_nearest_valid_voxels_empty_mask_error():
    """Test behavior when no valid points exist (should raise ValueError from KDTree)."""
    mask = np.zeros((5, 5), dtype=bool)
    dof_inds = np.array([[1, 1]])

    with pytest.raises(ValueError):
        mritk.data.orientation.find_nearest_valid_voxels(dof_inds, mask, k=1)
