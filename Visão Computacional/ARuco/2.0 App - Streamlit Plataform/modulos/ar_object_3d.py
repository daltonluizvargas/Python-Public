def render(cv2, np, math, img, obj, projection, colorr):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 0.5
    h, w = 300, 300

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + 150, p[1] + 150, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if colorr is False:
            cv2.fillConvexPoly(img, imgpts, (200, 27, 11))
        else:
            color = "".join(str(tem) for tem in face[-1])
            color = hex_to_rgb_v3(color)
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)
    return img


def projection_matrix(cv2, np, math, cam_parameters, homography):
    homography = homography * (-1)

    rot_and_trans = np.dot(np.linalg.inv(cam_parameters), homography)
    col_1 = rot_and_trans[:, 0]
    col_2 = rot_and_trans[:, 1]
    col_3 = rot_and_trans[:, 2]
    ii = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / ii
    rot_2 = col_2 / ii
    translation = col_3 / ii
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(cam_parameters, projection)


def hex_to_rgb_v3(hex_color):
    hex_color = hex_color.lstrip("#")
    h_len = len(hex_color)
    # Pad the hex_color with zeros if it has an incomplete length
    hex_color += "0" * (6 - h_len)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def hex_to_rgb_v4(hex_color):
    hex_color = hex_color.lstrip("#")
    h_len = len(hex_color)
    # Pad the hex_color with zeros if it has an incomplete length
    hex_color += "0" * (6 - h_len)
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    # Calculate shades of blue and ciano
    scale = 0.7  # You can adjust this scale to control the intensity of blue/ciano
    blue = int(b * scale)
    ciano = int(g * scale + (255 - g) * scale)
    return blue, ciano, ciano
