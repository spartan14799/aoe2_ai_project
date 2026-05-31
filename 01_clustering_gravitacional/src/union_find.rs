pub struct UnionFind {
    father: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self{
            father: (0..n).collect(),
            size: vec![1; n],
        }
    }

    pub fn find(&mut self, i: usize) -> usize{
        if self.father[i] == i{
            return i;
        }
        self.father[i] = self.find(self.father[i]);
        self.father[i]
    }
    pub fn union(&mut self, i: usize, j: usize){
        let root_i = self.find(i);
        let root_j = self.find(j);

        if root_i != root_j{
            if self.size[root_i] < self.size[root_j]{
                self.father[root_i] = root_j;
                self.size[root_j] += self.size[root_i];
            } else {
                self.father[root_j] = root_i;
                self.size[root_i] += self.size[root_j];
            }
        }
    }
}
