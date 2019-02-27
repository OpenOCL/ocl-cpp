classdef OclTensor < handle

  properties (Static)
  end

  properties
    idx % index number to identify instance
  end

  methods (Static)
    function r = create(idx)
      % Creates an OclTensor from an object that exists in the c program
      r = OclTensor();
      r.idx = idx;
    end
  end

  methods

    function self = OclTensor(dims)
      if nargin == 1
        % Construct the object in the c program
        self.idx = oclTensorMex('create', -1, dims);
      end
    end

    function r = uplus(self)
      idx = oclTensorMex('uplus', self.idx);
      r = OclTensor.create(idx);
    end

    function r = plus(self, other)
      idx = oclTensorMex('plus', self.idx, other.idx);
      r = OclTensor.create(idx);
    end

    function r = pow(self, exponent)
      idx = oclTensorMex('pow', self.idx, exponent);
      r = OclTensor.create(idx);
    end

    function r = str(self)
      r = oclTensorMex('str', self.idx);
    end

  end

end
