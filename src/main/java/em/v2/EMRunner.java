package em.v2;

public class EMRunner {

    public static void main(String[] args) {
        EMModel model = new EMModel();
        model.data = new EMModel.EStepData();
        EMModel modelCopy = model.copyForWorker();
        new A().cloneObj();
        // todo: add running of em sample
    }

    public static class A implements Cloneable {
        public A cloneObj () {
            try {
                Object o = super.clone();
                return (A) o;
            } catch(CloneNotSupportedException ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}
