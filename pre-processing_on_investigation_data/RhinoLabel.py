import rhinoscriptsyntax as rs
import scriptcontext as sc
import Rhino
# Virtual boreholes' labels are read in Rhino based on reference models
points = []
with open("..\data\points_12Boreholes.txt", 'r') as file:
    for line in file:
        points.append(line.strip('\n'))

objectIds = rs.GetObjects("Select soil bodies", rs.filter.polysurface)
Is_vail = True
Is_not_same = True
label = []
for j, xyz in enumerate(points):  # j: the number of points and xyz is the three-dimensional coordinate of a point
    for i, object_id in enumerate(objectIds):  # i: the stratigraphic number; object_id: the label of the original layer, i.e. the number of layers
        if rs.IsPointInSurface(object_id, xyz, strictly_in=False, tolerance=None):
            if Is_not_same:
                # If the point is inside the layer, the layer label counts and records that the point has taken effect;
                # If not, try comparing the next layer
                sc.doc = Rhino.RhinoDoc.ActiveDoc
                obj = sc.doc.Objects.Find(object_id)
                layer_id = obj.Attributes.LayerIndex
                label.append(str(layer_id) + "\n")
                Is_vail = False
                Is_not_same = False
    if Is_vail:
        label.append("-1" + "\n")  # Invalid points are marked as -1
    Is_vail = True
    Is_not_same = True
with open("..\data\points_12Boreholes_label.txt", "w") as file:
    file.writelines(label)

