#include <HPCortex.hpp>
#include <Testing.hpp>

/**
 * @brief Tests a simple linear pipeline by training a distributed neural network.
 *
 * This function tests the functionality of a pipelined neural network by comparing its performance with that of a non-pipelined version.
 * It first sets up a global pipeline across all ranks, generates synthetic data, defines models and costs,
 * trains both the pipelined and non-pipelined versions, and then compares their predictions.
 
void testSimpleLinearPipeline() {
 ...
}
 * @brief Enables global pipelining across all ranks.
 *
 * This line enables the distribution of computation across multiple ranks, allowing for parallel processing.
 
communicators().enableGlobalPipelining();
 * @brief Reports the setup of the communicator object.
 *
 * This line logs information about the current state of the communicator object after enabling global pipelining.
 
communicators().reportSetup();
 * @brief Retrieves the number of ranks in the pipeline.
 *
 * This variable stores the total count of ranks participating in the pipeline.
 
int nranks = communicators().pipelineNrank();
 * @brief Retrieves the rank ID of the current process.
 *
 * This variable identifies the unique position of the current process within the pipeline.
 
int rank = communicators().pipelineRank();
 * @brief Defines the batch size for individual calls.
 *
 * This parameter controls the number of samples processed together as a single unit during training.
 
int call_batch_size = 2;
 * @brief Specifies input and output dimensions for the pipeline block.
 *
 * These parameters define the shape of data exchanged between stages of the pipeline.
 
int in_out_dims[2] = {1, call_batch_size};
 * @brief Calculates the global batch size considering all ranks.
 *
 * The global batch size accounts for the combined sample capacity across all ranks in the pipeline.
 
int glob_batch_size = 6 * nranks;
 * @brief Sets the number of epochs for training.
 *
 * An epoch represents one complete pass through the entire dataset during the training phase.
 
int nepoch = 20;
 * @brief Determines the number of batches per epoch.
 *
 * Batches divide the dataset into smaller chunks for more efficient processing during training.
 
int nbatch = 10;
 * @brief Typedef for floating-point type used throughout calculations.
 *
 * Consistency in numerical representation ensures accuracy and precision in computations.
 
typedef float FloatType;
 * @brief Computes the total amount of data points generated.
 *
 * The product of batches per epoch and global batch size yields the overall dataset size.
 
int ndata = nbatch * glob_batch_size;
 * @brief Creates a vector to store pairs of input-output data points.
 *
 * Each pair consists of an input value (x) and its corresponding output (y).
 
std::vector<XYpair<FloatType, 1, 1>> data(ndata);
 * @brief Populates the data vector with synthesized values following a linear relationship.
 *
 * Input values (x) range from -1 to 1, while outputs (y) follow the equation y = 0.2x + 0.3.
 
for (int i = 0; i < ndata; i++) {
 ...
}
 * @brief Initializes weights and bias for the neural network layers.
 *
 * Weights and biases serve as learnable parameters that adjust during training to minimize prediction errors.
 
Matrix<FloatType> winit(1, 1, 0.1);
Vector<FloatType> binit(1, 0.01);
 * @brief Constructs a model for the current rank, incorporating either a linear layer or a ReLU-activated layer depending on the rank.
 *
 * Models at the last rank include an additional activation function to introduce non-linearity.
 
auto rank_model = rank == nranks - 1? enwrap(dnn_layer(input_layer<FloatType>(), winit, binit)) : enwrap(dnn_layer(input_layer<FloatType>(), winit, binit, ReLU<FloatType>()));
 * @brief Wraps the rank-specific model within a pipeline block for distributed execution.
 *
 * Pipeline blocks manage the flow of data through the model, adhering to specified input and output dimensions.
 
auto rank_block = pipeline_block<Matrix<FloatType>, Matrix<FloatType>>(rank_model, in_out_dims, in_out_dims);
 * @brief Defines a cost function wrapper around the pipeline block for evaluating loss during training.
 *
 * The cost function assesses the difference between predicted outputs and true labels, guiding the optimization process.
 
auto cost = BatchPipelineCostFuncWrapper<decltype(rank_block), MSEcostFunc<Matrix<FloatType>>>(rank_block, call_batch_size);
 * @brief Builds a comprehensive model encompassing all layers across ranks without pipelining.
 *
 * This full model serves as a reference for comparing the performance of the pipelined approach.
 
auto full_model = enwrap(dnn_layer(input_layer<FloatType>(), winit, binit));
for (int i = 0; i < nranks - 1; i++)
  full_model = enwrap(dnn_layer(std::move(full_model), winit, binit, ReLU<FloatType>()));
 * @brief Establishes a mean squared error cost function for the full model.
 *
 * This cost metric evaluates the average squared difference between predicted and actual outputs.
 
auto full_cost = mse_cost(full_model);
 * @brief Configures a learning rate scheduler with exponential decay.
 *
 * Learning rate scheduling adjusts the step size during gradient descent to achieve optimal convergence.
 
DecayScheduler<FloatType> lr(0.001, 0.1);
 * @brief Initializes parameters for the Adam optimizer.
 *
 * Adam adapts the learning rate for each parameter individually based on past gradients, enhancing stability and efficiency.
 
AdamParams<FloatType> ap;
 * @brief Instantiates the Adam optimizer with the configured learning rate schedule.
 *
 * The optimizer iteratively updates model parameters to minimize the defined cost function.
 
AdamOptimizer<FloatType, DecayScheduler<FloatType>> opt(ap, lr);
 * @brief Trains the pipelined model using the specified cost function, optimizer, and hyperparameters.
 *
 * Training involves iterative adjustments to model parameters to fit the observed data.
 
train(cost, data, opt, nepoch, glob_batch_size);
 * @brief Retrieves the optimized parameters of the trained model.
 *
 * Final parameters reflect the learned patterns and relationships within the training data.
 
Vector<FloatType> final_p = cost.getParams();
 * @brief Generates predictions using the trained model for all data points.
 *
 * Predictions represent the model's best estimates of output values given input data.
 
std::vector<Vector<FloatType>> predict(ndata);
for (int i = 0; i < ndata; i++)
  predict[i] = cost.predict(data[i].x);
 * @brief Disables parallelism for training the full model sequentially.
 *
 * Sequential training allows direct comparison with the pipelined approach under identical conditions.
 
std::cout << "Training rank local model for comparison" << std::endl;
communicators().disableParallelism();
communicators().reportSetup();
 * @brief Trains the full model using the same optimizer and hyperparameters as the pipelined model.
 *
 * This step provides a baseline for assessing the effectiveness of pipelining.
 
train(full_cost, data, opt, nepoch, glob_batch_size);
 * @brief Retrieves the optimized parameters of the fully trained model.
 *
 * These parameters serve as a reference point for validating the pipelined model's performance.
 
Vector<FloatType> expect_p = full_cost.getParams();
 * @brief Synchronizes processes before proceeding to ensure consistency across ranks.
 *
 * Barrier synchronization guarantees that all ranks have completed their assigned tasks before further operations.
 
MPI_Barrier(MPI_COMM_WORLD);
 * @brief Validates the pipelined model's performance against the full model's expectations.
 *
 * Comparison includes checking the closeness of optimized parameters and predicting capabilities.
 
if (!rank) {
 ...
}* This comment was generated by meta-llama/Llama-3.3-70B-Instruct:None at temperature 0.2.
*/ 
void testSimpleLinearPipeline(){
  //Test f(x) = 0.2*x + 0.3;
  communicators().enableGlobalPipelining(); //put all the ranks into a single pipeline
  communicators().reportSetup();
  
  int nranks = communicators().pipelineNrank();
  int rank = communicators().pipelineRank();

  int call_batch_size = 2;
  int in_out_dims[2] = {1,call_batch_size};
  
  int glob_batch_size = 6 * nranks;

  int nepoch = 20;
  int nbatch = 10;

  typedef float FloatType;
  
  int ndata = nbatch * glob_batch_size;
  std::vector<XYpair<FloatType,1,1> > data(ndata);

  for(int i=0;i<ndata;i++){
    FloatType eps = 2.0/(ndata - 1);
    FloatType x = -1.0 + i*eps; //normalize x to within +-1
    FloatType y = 0.2*x + 0.3;
    
    data[i].x = Vector<FloatType>(1,x);
    data[i].y = Vector<FloatType>(1,y);
  }
   
  Matrix<FloatType> winit(1,1,0.1);
  Vector<FloatType> binit(1,0.01);

  auto rank_model = rank == nranks-1 ? enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) )  : enwrap( dnn_layer(input_layer<FloatType>(), winit, binit, ReLU<FloatType>()) );
 
  auto rank_block = pipeline_block<Matrix<FloatType>, Matrix<FloatType> >(rank_model, in_out_dims, in_out_dims);

  auto cost = BatchPipelineCostFuncWrapper<decltype(rank_block), MSEcostFunc<Matrix<FloatType>> >(rank_block, call_batch_size);

  auto full_model = enwrap( dnn_layer(input_layer<FloatType>(), winit, binit) );
  for(int i=0;i<nranks-1;i++)
    full_model = enwrap( dnn_layer(std::move(full_model), winit, binit, ReLU<FloatType>()) );
  auto full_cost = mse_cost(full_model);

  DecayScheduler<FloatType> lr(0.001, 0.1);
  AdamParams<FloatType> ap;
  AdamOptimizer<FloatType,DecayScheduler<FloatType> > opt(ap,lr);

  //Train pipeline
  train(cost, data, opt, nepoch, glob_batch_size);
  Vector<FloatType> final_p = cost.getParams();
  std::vector<Vector<FloatType>> predict(ndata);
  for(int i=0;i<ndata;i++) predict[i] = cost.predict(data[i].x);

  std::cout << "Training rank local model for comparison" << std::endl;  
  communicators().disableParallelism();
  communicators().reportSetup();
  train(full_cost, data, opt, nepoch, glob_batch_size);
  Vector<FloatType> expect_p = full_cost.getParams();

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!rank){
    std::cout << "Final params " << final_p << " expect " << expect_p << std::endl;
    assert(near(final_p,expect_p,FloatType(1e-4),true));
    
    std::cout << "Predictions:" << std::endl;
    for(int i=0;i<nbatch;i++)
      std::cout << "Got " << predict[i] << " expect " << full_cost.predict(data[i].x) << " actual " << data[i].y << std::endl;
  }

}

/**
 * @brief Program entry point
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return Exit status of the program
 * This comment was generated by meta-llama/Llama-3.3-70B-Instruct:None at temperature 0.2.
*/ 
int main(int argc, char** argv){
  initialize(argc, argv);
  
  testSimpleLinearPipeline();

  return 0;
}
