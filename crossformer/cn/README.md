# READ THIS FIRST

hydra is a bit complicated to use, so I'll try to explain it here.

its great for structured config of experiments, but there are a lot of contraints which are not always immediately obvious

this repo has made some assumptions/tradeoffs to adjust hydra for my case

my priorities:
* experiment sweeps. run lots of experiments with different hparam easily
* config structure. clearly see what hparams are used and where
* modularity. easy to add new hparams and new models
* less boilerplate code.

what hydra offers that i dont care about:
* lots of yaml. python is usually cleaner i think 
    * (we might use yaml for sweeps if sweep classes arent possible, but only last resort)

## How to use

to add config node, build a class that is child of CN (registered via CNMeta)

CNMeta manages the registration of nodes and ConfigStore tree building 

all CN are also dataclass

### rules and things to consider about hydra

hydra nodes can only see other CN if they are childeren of the same parent node

ie: 
```
    /
        <the base config>
        data
            <all your configs for datasets>
            transform
                <all your configs for transforms>
        model
            <all your configs for models>
            head
                <all your configs for output heads>
            tokenizers
                <all your configs for input tokenizers>
```

also they have to use the same name for the attribute as they are registered under

ie: 

```python
class Run(CN):
    model: Model # cannot name this attr mymodel, since registered in model group
    data: Data # cannot name this attr dataset, since registered in data group
```

this is kind of annoying but thats the rule. 
* if you cant change the attr name to be like the group 
    * then you can double register the node with your preffered name
    * but not recommended since it will be confusing... also might break at some point

### defaults list

if you read the hydra docs you see that they require a defaults list in the config class

CNMeta builds this under the hood because it looks ugly and confusing in the nodes
* in short, it instantiates your class without the defaults 
* then retroactively looks for other config nodes
    * defines a new class with the defaults list added and returns this as the original class
* this forces the `__self__` attr of defaults to be at the end of defaults list
    * if you don't like that, dont use my code
    * or dont override the defaults

thats about it i think

### pathfinding

the current way of pathfinding is a bit annoying... we use a _group attr that is thrown away

if also tried looking at the file path to build the ConfigStore tree
* will revert back to this since it was better

```python
# given the following
path = 'path/from/repo_beginning_at_cn/group/subgroup/file_name_doesnt_matter.py'
ConfigStore.instance().store(name='any', node=MyNode, group='group/subgroup')
```

usually 
* base.py is the file with base/default config 
* other.py is some modification to the base node
