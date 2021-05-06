package em.v2;

import JSci.maths.matrices.AbstractDoubleMatrix;
import JSci.maths.matrices.DoubleSquareMatrix;
import JSci.maths.vectors.AbstractDoubleVector;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class Utils {

    public static <T> T[] init(T[] array, Supplier<? extends T> initializer) {
        for (int i = 0; i < array.length; i++) {
            array[i] = initializer.get();
        }
        return array;
    }

    public static int indexOfMaxInRow(AbstractDoubleMatrix matrix, int row) {
        double max = matrix.getElement(row, 0);
        int indexOfMax = 0;
        for (int i = 0; i < matrix.columns(); i++) {
            double el = matrix.getElement(row, i);
            if (el > max) {
                max = el;
                indexOfMax = i;
            }
        }

        return indexOfMax;
    }

    public static AbstractDoubleMatrix vectorSquare(AbstractDoubleVector vector) {
        int dim = vector.dimension();
        double[][] result = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                result[i][j] = vector.getComponent(i) * vector.getComponent(j);
            }
        }
        return new DoubleSquareMatrix(result);
    }

    public static <T> CompletableFuture<List<T>> sequence(List<CompletableFuture<T>> futures) {
        return CompletableFuture.allOf(futures.toArray(new CompletableFuture<?>[0]))
                .thenApply(v -> futures.stream()
                        .map(CompletableFuture::join)
                        .collect(Collectors.toList())
                );
    }
}
