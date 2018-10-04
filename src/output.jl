@recipe function f(grid::Grid{dim, T}, sol::NamedTuple{(:t, :points)}) where {dim, T}
    np = length(sol.points)
    points = Matrix{T}(undef, np, dim)
    for i in 1:np
        points[i,:] = sol.points[i].x
    end
    if dim ≥ 1
        xlims --> (minaxis(grid, 1), maxaxis(grid, 1))
        if dim == 1
            ylims --> (-1, 1)
        end
    end
    if dim ≥ 2
        ylims --> (minaxis(grid, 2), maxaxis(grid, 2))
    end
    if dim ≥ 3
        zlims --> (minaxis(grid, 3), maxaxis(grid, 3))
    end
    fillcolor := :blue
    seriestype := :scatter
    dim == 1 ? (vec(points), _ -> zero(T)) : ntuple(d -> @view(points[:,d]), Val(dim))
end
@recipe f(sol::Solution, t::Real) = (sol.grid, sol(t))
@recipe f(sol::Solution) = (sol.grid, sol[end])

#=
function output_vtk(filename::AbstractString, _sol::Solution{dim, T}, t::Real) where {dim, T}
    sol = _sol(t)
    np = length(sol.points)
    points = Matrix{T}(undef, dim, np)
    cells = fill(MeshCell(VTKCellTypes.VTK_VERTEX, [1]), np)
    for i in 1:np
        points[:,i] = sol.points[i].x
    end
    vtkfile = vtk_grid(filename, points, cells)
end
@inline write_vtk(filename::AbstractString, sol::Solution, t::Real) = vtk_save(output_vtk(filename, sol, t))

function output_vtk(filename::AbstractString, sol::Solution)
    mkpath("output")
    filename = "output/$filename"
    pvd = paraview_collection(filename)
    for i in 1:length(sol)
        t = sol.t[i]
        vtk = output_vtk(string(filename, "_", i), sol, t)
        vtk_save(vtk)
        collection_add_timestep(pvd, vtk, t)
    end
    return vtk_save(pvd)
end
=#
