class Polynomial:
    
    def __init__(self, coefficients):
        """ input: coefficients are in the form a_n, ...a_1, a_0 
        """
        self.coefficients =  coefficients # tuple is turned into a list
     
    def __repr__(self):
        """
        method to return the canonical string representation 
        of a polynomial.
   
        """
        return "Polynomial" + str(self.coefficients)
            
    def __call__(self, x):    
        res = 0
        for index, coeff in enumerate(self.coefficients):
            #print(index,coeff)
            res = res + coeff * x** index
        return res 


    def dot(self,x):
        res=0
        for index, coeff in enumerate(self.coefficients):
                if index > 0 :
                    res = res + (index)*coeff * x** (index-1)
        return res
