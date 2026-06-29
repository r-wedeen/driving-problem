importScripts("vendor/lapack_projection_bundle.js");

self.postMessage({ type: "ready" });

self.onmessage = (event) => {
  const { id, matrix, vector, rows, cols, rcond } = event.data;
  try {
    const projection = globalThis.CustomLapackProjection.projectLeastSquares(
      new Float64Array(matrix),
      new Float64Array(vector),
      rows,
      cols,
      rcond
    );
    self.postMessage({
      id,
      rank: projection.rank,
      update: projection.update.buffer
    }, [projection.update.buffer]);
  } catch (error) {
    self.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error)
    });
  }
};
