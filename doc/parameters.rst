.. _av_parameters_label:

Parameter overview
------------------
The software Olga and the library avtraj use text files in JavaScript Object Notation (JSON) as exchange data format
(Fig. 1). These JSON files:

1. define all necessary parameters for the calculation of accessible volumes (AVs)
2. store experimental data to validate models against experimentally determined distances
3. store information on necessary pre-processing of structural models for successful calculations of accessible volumes
4. instruct the software Olga to actions, e.g., saving accessible volumes to later visualization

.. literalinclude:: ./examples/labeling.fps.json
  :language: JSON


The top-level of these JSON files is a dictionary where the most relevant keys are

1. "Distances"
2. "Positions"
3. "version".

Via the keys “Positions” and “Distances” dictionaries can be accessed, which store the necessary information to
simulate and compare simulated and experimental distances, respectively (Fig. 1.). The value accessed by the "version"
key refers to the version of the JSON file.

The keys of the "Positions" dictionary serve as identifier of the labeling
positions, which are referred to be the position names in the “Distances” dictionary.


Distances
---------

.. list-table:: AV parameters: Distances
   :widths: 25 25 50 25 5 5 5
   :header-rows: 1

   * - Parameter, (type), optional
     - Options
     - Description
     - Example
     - FPS
     - Olga
     - avtraj
   * - distance_type (string), mandatory
     - “RDAMean“, “RDAMeanE”, “Rmp”, “Efficiency”
     - The type of distance that is calculated between for the set of labeling positions.
     - “RDAMean“
     - Same for all distances
     - \+
     - \+
   * - position1_name (string), mandatory
     -
     - The name of the first position. This name refers to the entry defined in the dictionary of labeling positions
     - “Labeling_site_A”
     - \+
     - \+
     - \+
   * - position2_name (string), mandatory
     -
     - The name of the second position. This name refers to the entry defined in the dictionary of labeling positions
     - “LP123”
     - \+
     - \+
     - \+
   * - distance (float), mandatory
     -
     - The reference distance (typically the experimental distance).
     - 45.5
     - \+
     - \+
     - \+
   * - error_neg (float), mandatory
     -
     - "error_pos" applies if Model – Data > 0 (see description "error_neg").
     - 1.2
     - \+
     - \+
     - \+
   * - error_pos (float), mandatory
     -
     - "error_pos" applies if Model – Data > 0 (see description "error_neg").
     - 1.4
     - \+
     - \+
     - \+
   * - Forster_radius (float), mandatory for the distance_type options “Efficiency” and “RDAMeanE”
     -
     - The Forster radius of the dye pair selected by position1_name and position2_name.
     - 51.5
     - Same for all distances
     - \+
     - \+
   * - distance_samples (int), optional
     -
     - Optional parameter to set the number of random distance samples to calculate the distance between the two positions. Default value 200000
     - 10000
     - Same for all distances
     - \+
     - \+
   * - distance_sampling_method (string), optional
     - "sobol_sequence", "random" (default), "weighted_random"
     - All distances are weighted by the combined probability. "random": Calculates distances between random points (NOT taken according their weight) in the AVs. "sobol": Calculates for distances between grid points from Sobol-sequence.
     -
     - \-
     - \-
     - \+

Positions
---------

