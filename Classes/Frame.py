class Frame():

    def __init__(self, rgb, depth, mask, cx, cy, fx, fy, pose):
        self.rgb = rgb
        self.depth = depth
        self.mask = mask
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.pose = pose