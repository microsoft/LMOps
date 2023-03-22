
from evaluation.decomposition import Decomposition, draw_decomposition_graph
from evaluation.graph_matcher import AStarSearcher


examples = [
    # 0
    (Decomposition(["representatives from New York state or Indiana state",
                    "the life spans of @@1@@"]),
     Decomposition(["representatives from new york state",
                    "representatives from indiana state",
                    "@@1@@ or @@2@@",
                    "life spans of @@3@@"])),
    # 1
    (Decomposition(["the team owned by Jerry Jones",
                    "the 1996 coach of @@1@@"]),
     Decomposition(["the team owned by Jerry Jones",
                    "the 1996 coach of @@1@@"])),
    # 2
    (Decomposition(["the team with Baltimore Fight Song",
                    "year did @@1@@ win the Superbowl"]),
     Decomposition(["the team with Baltimore Fight Song",
                    "what year did @@1@@ win the Superbowl"])),
    # 3
    (Decomposition(["a us air flight",
                    "@@1@@ from toronto to san diego",
                    "@@2@@ with a stopover in denver"]),
     Decomposition(["us air flights",
                    "@@1@@ from toronto",
                    "@@2@@ to san diego",
                    "@@3@@ with a stopover in denver"])),
    # 4
    (Decomposition(["flights",
                    "@@1@@ from baltimore",
                    "@@2@@ to atlanta",
                    "@@3@@ that arrive before noon and i'd like to see flights",
                    "@@4@@ from denver",
                    "@@5@@ to atlanta",
                    "@@6@@ that arrive before noon"]),
     Decomposition(["flights from baltimore",
                    "@@1@@ to atlanta",
                    "@@2@@ that arrive before noon",
                    "flights from denver",
                    "@@4@@ to atlanta",
                    "@@5@@ that arrive before noon",
                    "@@3@@, @@6@@"])),
    # 5
    (Decomposition(["the club \"Bootup Baltimore\"",
                    "all the female members of @@1@@",
                    "the first name and last name for @@2@@"]),
     Decomposition(["all female members of the club bootcup baltimore",
                    "the first name and last name of @@1@@"])),

    # 6
    (Decomposition(["conferences,",
                    "the number of @@1@@",
                    "@@2@@ which have more than 60 papers ",
                    "@@3@@ containing keyword \" Relational Database \""]),
     Decomposition(["papers containing keyword \"relational databases\"",
                    "conferences which have more than 60 @@1@@",
                    "the number of @@2@@"])),

    # 7
    (Decomposition(["the movie released in the year 2000 or earlier",
                    "the title and director of @@1@@",
                    "worldwide gross",
                    "@@2@@ that had the highest @@3@@"]),
     Decomposition(["movies released in the year 2000 or earlier",
                    "@@1@@ that had the highest worldwide gross",
                    "the title and director of @@2@@"])),

    # 8
    (Decomposition(["team Tim Howard playing for it",
                    "@@1@@ owned by Malcolm Glazer"]),
     Decomposition(["the teams owned by Malcolm Glazer",
                    "@@1@@ that has Tim Howard playing for it"])),

    # 9
    (Decomposition(["the parties",
                    "@@1@@ that have both representatives in New York state"
                    " and representatives in Pennsylvania state"]),
     Decomposition(["representatives in new york state",
                    "representative in pennsylvania state",
                    "the parties of both @@1@@ and @@2@@"]))
]

searcher = AStarSearcher()

for i in range(len(examples)):
    if i < 9:
        continue

    dec1, dec2 = examples[i]
    graph1 = dec1.to_graph()
    graph2 = dec2.to_graph()

    # draw_decomposition_graph(graph1, title="prediction")
    # draw_decomposition_graph(graph2, title="gold")

    searcher.set_graphs(graph1, graph2)
    # res = searcher.a_star_search(debug=True)
    res12 = searcher.a_star_search(debug=False)

    searcher.set_graphs(graph2, graph1)
    res21 = searcher.a_star_search(debug=False)

    print("\nexample {}".format(i))
    for (res, desc) in [(res12, "1--2"), (res21, "2--1")]:
        print("edit path {}: ".format(desc), res[0])
        print("edit path string {}: ".format(desc), res[1])
        print("cost {}: ".format(desc), res[2])
        # print("normalized cost {}: ".format(desc), res[3])
