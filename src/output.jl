@recipe function f(snap::SnapShot{dim, T}) where {dim, T}
    np = length(snap)
    points = Matrix{T}(undef, np, dim)
    for i in 1:np
        points[i,:] = snap[i].x
    end
    if dim ≥ 1
        xlims --> snap.limits[1]
        if dim == 1
            ylims --> (-1, 1)
        end
    end
    if dim ≥ 2
        ylims --> snap.limits[2]
    end
    if dim ≥ 3
        zlims --> snap.limits[3]
    end
    markercolor --> [:blue]
    markerstrokewidth --> 0
    dpi --> 150
    seriestype := :scatter
    aspectratio := :equal
    dim == 1 ? (vec(points), _ -> zero(T)) : ntuple(d -> @view(points[:,d]), Val(dim))
end
@recipe f(sol::Solution, t::Real) = (sol(t),)
@recipe f(sol::Solution) = (sol[end],)

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
