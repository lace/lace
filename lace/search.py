
class MeshMixin(object):
    def compute_aabb_tree(self):
        from lace_search.aabb_tree import AabbTree
        return AabbTree(self.v, self.f)

    def compute_aabb_normals_tree(self):
        from lace_search.aabb_normals_tree import AabbNormalsTree
        return AabbNormalsTree(self)

    def compute_closest_point_tree(self, use_cgal=False):
        from lace_search.closest_point_tree import ClosestPointTree
        if use_cgal:
            raise NotImplementedError('use_cgal should be False, CGALClosestPointTree has been removed')
        return ClosestPointTree(self)

    def closest_vertices(self, vertices, use_cgal=False):
        if use_cgal:
            raise NotImplementedError('use_cgal should be False, CGALClosestPointTree has been removed')
        return self.compute_closest_point_tree().nearest(vertices)

    def closest_points(self, vertices):
        return self.closest_faces_and_points(vertices)[1]

    def closest_faces_and_points(self, vertices):
        return self.compute_aabb_tree().nearest(vertices)

    def vertices_within(self, vertex_or_vertices, radius, use_cgal=False):
        if use_cgal:
            raise NotImplementedError('use_cgal should be False, CGALClosestPointTree has been removed')
        tree = self.compute_closest_point_tree()
        return tree.vertices_within(vertex_or_vertices, radius)
