import torch
import torchvision.transforms.functional as F


def _get_flat_grid_locations(height, width, device):
    y_range = torch.linspace(0, height - 1, height, device=device)
    x_range = torch.linspace(0, width - 1, width, device=device)
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")
    return torch.stack((y_grid, x_grid), -1).reshape([height * width, 2])


def _phi(r, order):
    EPSILON = torch.tensor(1e-10, device=r.device)
    if order == 1:
        r = torch.max(r, EPSILON)
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.max(r, EPSILON))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.max(r, EPSILON))
    elif order % 2 == 0:
        r = torch.max(r, EPSILON)
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.max(r, EPSILON)
        return torch.pow(r, 0.5 * order)


def _cross_squared_distance_matrix(x, y):
    x_norm_squared = torch.sum(torch.mul(x, x))
    y_norm_squared = torch.sum(torch.mul(y, y))

    x_y_transpose = torch.matmul(x.squeeze(0), y.squeeze(0).transpose(0, 1))

    squared_dists = x_norm_squared - 2 * x_y_transpose + y_norm_squared

    return squared_dists.float()


def _solve_interpolation(pts, vals, order, eps: float = 1e-7) -> tuple:
    device = pts.device
    channels, n, d = F.get_dimensions(pts)
    k = vals.shape[-1]

    c = pts[0]
    f = vals[0].float()

    matrix_a = _phi(_cross_squared_distance_matrix(c, c), order)  # [n, n]
    ones = torch.ones(n, dtype=pts.dtype, device=device).view([n, 1])  # [n ,1]
    matrix_b = torch.cat((c, ones), -1).float()  # [n , d + 1]

    # [n + d + 1, n]
    left_block = torch.cat((matrix_a, torch.transpose(matrix_b, 1, 0)), 0)

    num_b_cols = matrix_b.shape[-1]

    lhs_zeros = torch.randn((num_b_cols, num_b_cols), device=device) * eps
    right_block = torch.cat((matrix_b, lhs_zeros), 0)  # [n + d + 1, d + 1]
    lhs = torch.cat((left_block, right_block), 1)  # [n + d + 1, n + d + 1]

    rhs_zeros = torch.zeros((d + 1, k), dtype=pts.dtype, device=device).float()
    rhs = torch.cat((f, rhs_zeros), 0)  # [n + d + 1, k]

    try:
        X = torch.linalg.solve(lhs, rhs)
        w = X[:n, :]
        v = X[n:, :]
        return w, v
    except torch._C._LinAlgError:
        return None, None


def _apply_interpolation(query_pts, pts, w, v, order):
    query_pts = query_pts.unsqueeze(0)
    pairwise_dists = _cross_squared_distance_matrix(
        query_pts.float(), pts.float()
    )
    phi_pairwise_dists = _phi(pairwise_dists, order)

    rbf_term = torch.matmul(phi_pairwise_dists, w)

    ones = torch.ones_like(query_pts[..., :1])
    query_points_pad = torch.cat((query_pts, ones), 2).float()
    linear_term = torch.matmul(query_points_pad, v)

    return rbf_term + linear_term


def _interpolate_spline(
    pts,
    vals,
    query_pts,
    order,
    regularization_weight=0.0,
) -> torch.Tensor:
    # First, fit the spline to the observed data.
    w, v = _solve_interpolation(pts, vals, order, regularization_weight)

    if w is None and v is None:
        return None

    # Then, evaluate the spline at the query locations.
    query_values = _apply_interpolation(query_pts, pts, w, v, order)

    return query_values


def _create_dense_flows(flattened_flows, height, width):
    # possibly .view
    return torch.reshape(flattened_flows, [height, width, 2])


def _interpolate_bilinear(
    grid, query_points, name="interpolate_bilinear", indexing="ij"
):
    if indexing != "ij" and indexing != "xy":
        raise ValueError("Indexing mode must be 'ij' or 'xy'")

    shape = grid.shape
    if len(shape) != 3:
        msg = "Grid must be 3 dimensional. Received size: "
        raise ValueError(msg + str(grid.shape))

    channels, height, width = F.get_dimensions(grid)

    shape = [height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype
    grid_device = grid.device

    num_queries = query_points.shape[0]

    alphas = []
    floors = []
    ceils = []
    index_order = [0, 1] if indexing == "ij" else [1, 0]
    unstacked_query_points = query_points.unbind(1)

    for dim in index_order:
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim]

        max_floor = torch.tensor(
            size_in_indexing_dimension - 2,
            dtype=query_type,
            device=grid_device,
        )
        min_floor = torch.tensor(0, dtype=query_type, device=grid_device)
        maxx = torch.max(min_floor, torch.floor(queries))
        floor = torch.min(maxx, max_floor)
        int_floor = floor.long()
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        alpha = (queries - floor).clone().detach().type(grid_type)
        min_alpha = torch.tensor(0, dtype=grid_type, device=grid_device)
        max_alpha = torch.tensor(1, dtype=grid_type, device=grid_device)
        alpha = torch.min(torch.max(min_alpha, alpha), max_alpha)

        alpha = torch.unsqueeze(alpha, 1)
        alphas.append(alpha)

    # work with only one channel
    flattened_grid = torch.reshape(grid[0], [height * width, 1])

    def gather(y_coords, x_coords, name):
        linear_coordinates = y_coords * width + x_coords
        gathered_values = torch.gather(
            flattened_grid.t(), 1, linear_coordinates.unsqueeze(0)
        )
        return torch.reshape(gathered_values, [num_queries, 1])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], "top_left")
    top_right = gather(floors[0], ceils[1], "top_right")
    bottom_left = gather(ceils[0], floors[1], "bottom_left")
    bottom_right = gather(ceils[0], ceils[1], "bottom_right")

    interp_top = alphas[1] * (top_right - top_left) + top_left
    interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
    interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    interp = interp.repeat(channels, 1, 1)

    return interp


def _dense_image_warp(tensor, flow):
    channels, height, width = F.get_dimensions(tensor)
    device = tensor.device

    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="ij",
    )

    stacked_grid = torch.stack((grid_y, grid_x), dim=2).float()
    b_grid = stacked_grid.unsqueeze(-1).permute(3, 1, 0, 2)
    query_points_on_grid = b_grid - flow

    query_points_flattened = torch.reshape(
        query_points_on_grid, [height * width, 2]
    )

    interpolated = _interpolate_bilinear(tensor, query_points_flattened)
    interpolated = torch.reshape(interpolated, [channels, height, width])

    return interpolated


def _sparse_image_warp(
    tensor,
    src_ctr_pt_locations,
    dest_ctr_pt_locations,
    order=2,
    regularization_weight=0.0,
):
    device = tensor.device
    control_point_flows = dest_ctr_pt_locations - src_ctr_pt_locations

    _, height, width = F.get_dimensions(tensor)
    flattened_grid_locations = _get_flat_grid_locations(height, width, device)

    flattened_flows = _interpolate_spline(
        dest_ctr_pt_locations,
        control_point_flows,
        flattened_grid_locations,
        order,
        regularization_weight,
    )

    if flattened_flows is None:
        return tensor, None

    dense_flows = _create_dense_flows(flattened_flows, height, width)

    warped_image = _dense_image_warp(tensor, dense_flows)
    return warped_image, dense_flows
