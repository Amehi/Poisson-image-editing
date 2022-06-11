import cv2
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

class PoissonEditor:
    def __init__(self, source_path, target_path, mask_path):
        self.source_img = cv2.imread(source_path)
        self.target_img = cv2.imread(target_path)
        self.mask_img = cv2.imread(mask_path, 0)
        self.mask = np.atleast_3d(self.mask_img).astype(np.single) / 255.
        self.mask[self.mask != 1] = 0
        self.mask = self.mask[:,:,0]
        self.channel_num = self.source_img.shape[-1]
        print("start")
        self.result_stack = [self.poissonEdit(self.source_img[:,:,i], self.target_img[:,:,i], self.mask) for i in range(self.channel_num)]
        self.result = cv2.merge(self.result_stack)
        cv2.imwrite('result.png', self.result)
        print("success")

    def neighbors(self, index):
        i,j = index
        return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]    

    def div(self, source, index):
        i,j = index
        return 1 * source[i+1, j] + 1 * source[i-1, j] + 1 * source[i, j+1] + 1 * source[i, j-1] - 4 * source[i, j]

    def pointOnEdge(self, mask, index):
        if(mask[index] == 1):
            for neighbor in self.neighbors(index):
                if(mask[neighbor] != 1):
                    return True
        return False
    
    def poissonEdit(self, source, target, mask):
        nonzero = np.nonzero(mask)
        points = list(zip(nonzero[0], nonzero[1]))
        n = len(points)
        print("process A")
        A = scipy.sparse.lil_matrix((n, n))
        #print(A.data)
        for i, index in enumerate(points):
            A[i,i] = -4
            for x in self.neighbors(index):
                if x not in points:
                    continue
                j = points.index(x)
                #print(A[i])
                A[i,j] = 1

        print("process b")
        b = np.zeros(n)
        for i, index in enumerate(points):
            b[i] = self.div(source, index)
            if(self.pointOnEdge(mask, index)):
                for neighbor in self.neighbors(index):
                    if (mask[neighbor] != 1):
                        b[i] -= target[neighbor]
        print("calculate x")
        x = scipy.sparse.linalg.cg(A, b)
        result = np.copy(target).astype(int)
        for i, index in enumerate(points):
            result[index] = x[0][i]
        return result

if __name__ == '__main__':
    source = "source.jpg"
    target = "target.jpg"
    mask = "mask.jpg"
    poisson = PoissonEditor(source, target, mask)