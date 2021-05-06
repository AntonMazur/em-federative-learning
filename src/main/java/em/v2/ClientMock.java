package em.v2;

import JSci.maths.vectors.AbstractDoubleVector;

public class ClientMock extends EMWorker {

    public ClientMock(AbstractDoubleVector[] data, int nClusters) {
        super(data, nClusters);
    }

    public ClientMock() {
        super(new AbstractDoubleVector[0], 0);
    }
}
