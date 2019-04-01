import DFControl: Point, Point3, Vec3, Mat3, Mat4

Point{N, T}(x::T) where {N, T} = Point{N, T}(x, x, x)
