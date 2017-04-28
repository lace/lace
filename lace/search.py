
class MeshMixin(object):

    msg_lace_search_missing = "{} is implemented in `lace-search` which is not installed and not yet publicly avaliable"

    def compute_aabb_tree(self):
        try:
            from lace_search.aabb_tree import AabbTree
        except ImportError:
            raise NotImplementedError(self.msg_lace_search_missing.format('compute_aabb_tree'))
        return AabbTree(self.v, self.f)

    def compute_aabb_normals_tree(self):
        try:
            from lace_search.aabb_normals_tree import AabbNormalsTree
        except ImportError:
            raise NotImplementedError(self.msg_lace_search_missing.format('compute_aabb_normals_tree'))
        return AabbNormalsTree(self)

    def compute_closest_point_tree(self):
        try:
            from lace_search.closest_point_tree import ClosestPointTree
        except ImportError:
            raise NotImplementedError(self.msg_lace_search_missing.format('compute_closest_point_tree'))
        return ClosestPointTree(self)

    def closest_vertices(self, vertices):
        return self.compute_closest_point_tree().nearest(vertices)

    def closest_points(self, vertices):
        return self.closest_faces_and_points(vertices)[1]

    def closest_faces_and_points(self, vertices):
        return self.compute_aabb_tree().nearest(vertices)

    def vertices_within(self, vertex_or_vertices, radius):
        tree = self.compute_closest_point_tree()
        return tree.vertices_within(vertex_or_vertices, radius)
