var map3 = function(fun, args1, args2, args3){
  var fun2 = function(arg12, arg3) { return fun(arg12[0], arg12[1], arg3)}
  var args12 =zip(args1, args2);
  print(args12);
  return map2(fun2, args12, args3);
};

var max = function(a,b){
  return Math.max(a,b);
};


var actions = ['A', 'B'];
var startStates = ['start1','start2'];
var nextStates = ['A', 'B'];


var transition = function(state, action) {
  var nextProbs = (action === 'A') ? [1,0] : [0,1];
  return Categorical({ps:nextProbs, vs:nextStates});
};

//distributions of true parameters and robot's belief
var TRUED = Categorical({ps:[1,1], vs:[-1,1]});
var TRUEBETA1 = Categorical({ps:[1,1,1], vs:[1,2,3]});
var TRUEBETA2 = Categorical({ps:[1,1,1], vs:[4,2,3]});

var TRUEUTILA = 20;


var getTrueBetaDist = function(state){
    var table = {
      start1: TRUEBETA1,
      start2: TRUEBETA2
    };
    return table[state];
};


  var getTrueUtility = function(state, d){
    var table = {
      A: TRUEUTILA,
      B: TRUEUTILA + d
    };
    return table[state];
  };

var getUtilities = function(d){
 var dFun = function(n){return d;};
 // create an array of length(nextStates) ds
 var n = nextStates.length;
 var ds = mapN(dFun, n);
 return map2(getTrueUtility, nextStates, ds);
};

var expectedUtility = function(state, action, d) {
       return expectation(Infer({ model() { 
         var current = state;
         //print(transition(current, action));
         var next = sample(transition(current, action));
         var util = getTrueUtility(next,d);
         return util;
       }}));
};


var softMaxAgent = function(state, beta, d) {
      return Infer({ method:"enumerate",
        model() {
          var action = uniformDraw(actions);
          var eu = expectedUtility(state, action, d);
          var debugPrint = false;
          if (debugPrint){
            print("action, state, beta, d, eu =");
            print(action);
            print(state);
            print(beta);
            print(d);
            print(eu);
            print("factor");
            print(eu/beta);
          }
          
          factor(eu/beta);
          
        return action;
        }
      });

};

//generate sample trajectory
var makeTrajectory = function(length, d, getBeta) {
  var step = function(){
    var state = uniformDraw(startStates);
    var beta = getBeta(state);
    var action = sample(softMaxAgent(state, beta, d));
    return [state, action];
  };
  var res = step()
  return length==1 ? [res] : [res].concat(makeTrajectory(length-1, d, getBeta));
};


var posterior = function(observedTrajectory){
  return Infer({model(){
    //sample possible values of d, beta1, beta2
    var d = sample(TRUED);
    var beta1 = sample(TRUEBETA1);
    var beta2 = sample(TRUEBETA2);
    var getBeta = function(state){
      var table = {
        start1: beta1,
        start2: beta2
      };
    return table[state];
    };

    // For each observed state-action pair, factor on likelihood of action
    map(
      function(stateAction){
        var state = stateAction[0];
        var beta = getBeta(state);
        var action = stateAction[1];
        observe(softMaxAgent(state, beta, d), action);
      },
      observedTrajectory);
    return d;
  }});
};

//model the regret of the human based on their beta level
var humanScore = function(state, d, beta) {  
  Infer( { model(){
    var action = sample(softMaxAgent(state, beta, d));
    var nextState = sample(transition(state, action));
    var actualUtility = getTrueUtility(nextState, d);
    var utilities = getUtilities(d);
    var maxUtility = reduce(max, -999999, utilities);
    var regret = actualUtility - maxUtility;
    var debug = false;
    if (debug) {
      print("action, nextState, actual, max, regret=");
      print(action);
      print(nextState);
      print(actualUtility);
      print(maxUtility);
      print(regret);
    };
    return regret;
   }});
};


var robotScore = function(length, d, getBeta){
  Infer({model() {   
    var observedTrajectory = makeTrajectory(length, d, getBeta);
    var posteriorD = posterior(observedTrajectory);  
    var estimateD = expectation(posteriorD);
    var correctChoice = (d*estimateD) >0; //true if d and estimateD have same sign
    var regret = correctChoice ? 0 : -Math.abs(d);
    var debug = false;
    if (debug) {
      print("posterior, correct, estimateD, regret")
      print(posteriorD);
      print(correctChoice);
      print(estimateD);
      print(regret);    
    };
    return regret;
  }});
};


  //returns counts of how many times each action appears in a trajectory
 var getCounts = function(traj) {
   var count = function(target){
      var test = function(stateActionPair) {
        return (target == stateActionPair[1]) ? 1 : 0
      }
      var flags = map(test, traj); //array containing a 1 for each match
      return sum(flags);
    }
    var counts = map(count, actions);
    //print(counts);
    return counts;  
 };
  
  

var naiveRobotScore = function(length, d, getBeta){
  //Naive robot chooses the most frequent action from the observed traj
  Infer({model() {  
    var observedTrajectory = makeTrajectory(length, d, getBeta);
    var counts = getCounts(observedTrajectory);

    var frequencyDifference = counts[1]-counts[0];
    if (frequencyDifference==0) {
      //in this case, the two actions have equal frequencies 
      //and robot chooses at random
      var regret = -0.5*Math.abs(d);
      //print(regret);
      return {regret,length};
    } 
    else{ //robot chooses most frequent action
      var correctChoice = d*frequencyDifference >0; 
      //true (robot chooses correctly) if d and frequencyDifference have same sign
      var regret = correctChoice ? 0 : -Math.abs(d)
      //print(regret);
      return regret;
    }

  }});
};

var sampleBeta1 = 1;
var sampleBeta2 = 1;

var getSampleBeta = function(state){
    var table = {
      start1: sampleBeta1,
      start2: sampleBeta2
    };
    return table[state];
};

var simulateRobot = function (scorer, lengths, ds, beta1, beta2) {
  return expectation(Infer({ model(){
    var length = uniformDraw(lengths);
    var d = uniformDraw(ds);
    var getBeta = function(state){
      var table = {
        start1: sample(beta1),
        start2: sample(beta2)
      };
      return table[state];
    };
    var expectedScore = expectation(scorer(arg1, arg2, arg3));
    var debug = false;
    if (debug) {
      print("1,2,3, expectedScore =");
      print(arg1);
      print(arg2);
      print(arg3);
      print(expectedScore);
    }
    return expectedScore;
  }}));
};

var simulateHuman = function (scorer, states, ds, betas) {
  return Infer({ model(){
    var getBeta = function(state){
      var table = {
        start1: betas[0],
        start2: betas[1]
      };
      return table[state];
    };
    var state = uniformDraw(states);
    var beta = getBeta(state);
    var d = uniformDraw(ds);
    var expectedScore = expectation(scorer(state, d, beta));
    var debug = false;
    if (debug) {
      print("state, d, beta, expectedScore =");
      print(state);
      print(d);
      print(beta);
      print(expectedScore);
    }
    return expectedScore;
  }});
};


var humanRegret = humanScore('start1', 1, 1);
print(humanRegret);
viz(humanRegret);


//var scores = simulateHuman(humanScore, startStates, [1,2,3], [TRUEBETA1, TRUEBETA2]);
//print(scores);
//viz(scores);

var scores2 = map( 
  function([beta1, beta2]) { return expectation(simulateHuman(
    humanScore, 
    startStates, 
    [1],
    [beta1, beta2]
  ));},
  [[1,1],[2,2],[3,3],[4,4], [5,5], [6,6], [7,7]]);

print(scores2);
viz(scores2);
 
//map(viz,humanRegrets);
//var humanMeans = map(expectation, humanRegrets);
//var humanRegret = listMean(humanMeans);
//print(humanRegret);
  
//var naiveRobotRegret = naiveRobotScore(1, 1, getSampleBeta);
//print("naive");
//print(naiveRobotRegret);
//viz(naiveRobotRegret);

//map(viz, robotRegrets);
//viz(robotscore);
//map(viz, naiveRobotRegrets);