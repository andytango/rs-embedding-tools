use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::Array2;

pub mod hdbscan;
pub mod pca;
pub mod pacmap;

#[derive(Serialize, Deserialize)]
pub struct InputData {
    pub data: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct OutputData {
    pub clusters: Vec<Option<usize>>,
    pub reduced_data: Vec<Vec<f64>>,
}

#[wasm_bindgen]
pub fn process_embeddings(val: JsValue) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    let input: InputData = serde_wasm_bindgen::from_value(val)?;

    let number_of_datapoints = input.data.len();
    if number_of_datapoints == 0 {
        return Err(JsValue::from_str("Input data is empty"));
    }
    let dimensions = input.data[0].len();

    // Convert the data to an ndarray Array2<f64>
    let data_vec: Vec<f64> = input.data
        .into_iter()
        .flatten()
        .map(|x| x as f64)
        .collect();

    let data_array = Array2::from_shape_vec(
        (number_of_datapoints, dimensions),
        data_vec,
    ).map_err(|e| JsValue::from_str(&format!("Failed to convert to Array2: {}", e)))?;

    // Reduce dimensionality with PACMAP
    let pacmap = pacmap::PacmapBuilder::new()
        .n_components(2)
        .n_neighbors(5.min(number_of_datapoints - 1))
        .n_iterations(450)
        .build();

    let reduced_data_array = pacmap
        .fit_transform(&data_array)
        .map_err(|e| JsValue::from_str(&format!("PACMAP error: {}", e)))?;

    // Perform clustering with HDBSCAN
    let clusterer = hdbscan::HdbscanBuilder::new()
        .min_cluster_size(5)
        .min_samples(5)
        .build();

    let clusters_i32 = clusterer
        .fit_predict(&reduced_data_array)
        .map_err(|e| JsValue::from_str(&format!("HDBSCAN error: {}", e)))?;

    // Convert Vec<i32> to Vec<Option<usize>>
    let clusters: Vec<Option<usize>> = clusters_i32
        .into_iter()
        .map(|c| if c == -1 { None } else { Some(c as usize) })
        .collect();

    // Convert the reduced data back to a Vec<Vec<f64>>
    let reduced_data: Vec<Vec<f64>> = reduced_data_array
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect();

    let output = OutputData {
        clusters,
        reduced_data,
    };

    Ok(serde_wasm_bindgen::to_value(&output)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;
    use wasm_bindgen_test::wasm_bindgen_test_configure;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_process_embeddings() {
        let input_data = InputData {
            data: vec![
                vec![1.0, 2.0, 3.0],
                vec![1.1, 2.1, 3.1],
                vec![10.0, 11.0, 12.0],
                vec![10.1, 11.1, 12.1],
                vec![100.0, 101.0, 102.0],
                vec![100.1, 101.1, 102.1],
                vec![100.2, 101.2, 102.2],
            ],
        };

        let js_value = serde_wasm_bindgen::to_value(&input_data).unwrap();
        let result = process_embeddings(js_value).unwrap();
        let output: OutputData = serde_wasm_bindgen::from_value(result).unwrap();

        // Basic assertions
        assert_eq!(output.clusters.len(), input_data.data.len());
        assert_eq!(output.reduced_data.len(), input_data.data.len());
        assert_eq!(output.reduced_data[0].len(), 2); // PACMAP reduces to 2 dimensions

        // Check if all reduced data points are finite
        for row in output.reduced_data.iter() {
            for &val in row.iter() {
                assert!(val.is_finite(), "Reduced data contains non-finite values");
            }
        }
    }
}
