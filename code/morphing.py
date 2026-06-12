import numpy as np
import imageio.v2 as imageio
from scipy.spatial import Delaunay
import os



def load_points(filename):

    return np.loadtxt(filename)



def affine_matrix(src_tri, dst_tri):


    A = np.array([
        [src_tri[0,0], src_tri[0,1], 1],
        [src_tri[1,0], src_tri[1,1], 1],
        [src_tri[2,0], src_tri[2,1], 1]
    ])

    Bx = dst_tri[:,0]
    By = dst_tri[:,1]

    ax = np.linalg.solve(A, Bx)
    ay = np.linalg.solve(A, By)

    M = np.array([
        [ax[0], ax[1], ax[2]],
        [ay[0], ay[1], ay[2]],
        [0,0,1]
    ])

    return M


def point_in_triangle(pt, tri):

    x,y = pt

    x1,y1 = tri[0]
    x2,y2 = tri[1]
    x3,y3 = tri[2]

    det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)

    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
    c = 1 - a - b

    return (a >= 0) and (b >= 0) and (c >= 0)



def morph(img1, img2, img1_pts, img2_pts, tri, warp_frac, dissolve_frac):

    h, w, c = img1.shape
    result = np.zeros_like(img1)

    inter_pts = (1 - warp_frac)*img1_pts + warp_frac*img2_pts

    for triangle in tri.simplices:

        t1 = img1_pts[triangle]
        t2 = img2_pts[triangle]
        t  = inter_pts[triangle]

        M1 = affine_matrix(t, t1)
        M2 = affine_matrix(t, t2)

        xmin = int(np.floor(np.min(t[:,0])))
        xmax = int(np.ceil(np.max(t[:,0])))
        ymin = int(np.floor(np.min(t[:,1])))
        ymax = int(np.ceil(np.max(t[:,1])))

        xmin = max(xmin,0)
        ymin = max(ymin,0)
        xmax = min(xmax,w-1)
        ymax = min(ymax,h-1)

        for y in range(ymin, ymax+1):
            for x in range(xmin, xmax+1):

                if point_in_triangle((x,y), t):

                    p = np.array([x,y,1])

                    src1 = M1 @ p
                    src2 = M2 @ p

                    x1,y1 = int(src1[0]), int(src1[1])
                    x2,y2 = int(src2[0]), int(src2[1])

                    if (0 <= x1 < w and 0 <= y1 < h and
                        0 <= x2 < w and 0 <= y2 < h):

                        color1 = img1[y1,x1]
                        color2 = img2[y2,x2]

                        result[y,x] = (
                            (1 - dissolve_frac) * color1 +
                            dissolve_frac * color2
                        )

    return result.astype(np.uint8)

def main():

    img1 = imageio.imread("PS4.png")
    img2 = imageio.imread("Xbox.png")

    pts1 = load_points("PS4.txt")
    pts2 = load_points("Xbox.txt")

    h, w, _ = img1.shape

    corners = np.array([
        [0,0],
        [w-1,0],
        [w-1,h-1],
        [0,h-1]
    ])

    pts1 = np.vstack([pts1, corners])
    pts2 = np.vstack([pts2, corners])

    mean_pts = (pts1 + pts2) / 2

    tri = Delaunay(mean_pts)

    os.makedirs("frames", exist_ok=True)

    n_frames = 60

    for i in range(n_frames):

        f = i / (n_frames - 1)

        frame = morph(
            img1,
            img2,
            pts1,
            pts2,
            tri,
            warp_frac=f,
            dissolve_frac=f
        )

        filename = f"frames/file_{i:05d}.png"

        imageio.imwrite(filename, frame)

        print("Saved", filename)


if __name__ == "__main__":
    main()