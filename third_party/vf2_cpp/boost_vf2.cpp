#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/vf2_sub_graph_iso.hpp>
#include <boost/property_map/property_map.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

struct VertexProp {
    int label;
};

struct EdgeProp {
    int label;
};

using Graph = boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::undirectedS,
    VertexProp,
    EdgeProp
>;

using EdgeTuple = std::tuple<int, int, int>;

static Graph build_graph(const std::vector<int>& node_labels, const std::vector<EdgeTuple>& edges) {
    Graph graph(node_labels.size());
    for (std::size_t i = 0; i < node_labels.size(); ++i) {
        graph[static_cast<Graph::vertex_descriptor>(i)].label = node_labels[i];
    }
    for (const auto& [src, dst, label] : edges) {
        auto edge = boost::add_edge(src, dst, graph);
        if (edge.second) {
            graph[edge.first].label = label;
        }
    }
    return graph;
}

struct AlwaysFalseCallback {
    template <typename CorrespondenceMap1To2, typename CorrespondenceMap2To1>
    bool operator()(CorrespondenceMap1To2, CorrespondenceMap2To1) const {
        return false;
    }
};

struct CollectNodeSetsCallback {
    const Graph& query;
    std::set<std::vector<int>>& node_sets;
    std::size_t max_node_sets;

    template <typename CorrespondenceMap1To2, typename CorrespondenceMap2To1>
    bool operator()(CorrespondenceMap1To2 query_to_target, CorrespondenceMap2To1) const {
        std::vector<int> node_set;
        node_set.reserve(boost::num_vertices(query));
        auto vertices = boost::vertices(query);
        for (auto it = vertices.first; it != vertices.second; ++it) {
            node_set.push_back(static_cast<int>(boost::get(query_to_target, *it)));
        }
        std::sort(node_set.begin(), node_set.end());
        node_sets.insert(std::move(node_set));
        return max_node_sets == 0 || node_sets.size() < max_node_sets;
    }
};

struct Vf2Timeout {};

static bool contains_subgraph(
    const std::vector<int>& query_node_labels,
    const std::vector<EdgeTuple>& query_edges,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges) {
    py::gil_scoped_release release;
    Graph query = build_graph(query_node_labels, query_edges);
    Graph target = build_graph(target_node_labels, target_edges);

    auto vertex_comp = [&query, &target](auto query_v, auto target_v) {
        return query[query_v].label == target[target_v].label;
    };
    auto edge_comp = [&query, &target](auto query_e, auto target_e) {
        return query[query_e].label == target[target_e].label;
    };

    return boost::vf2_subgraph_mono(
        query,
        target,
        AlwaysFalseCallback{},
        boost::vertex_order_by_mult(query),
        boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
    );
}

static std::vector<std::vector<int>> node_sets_subgraph_mono(
    const std::vector<int>& query_node_labels,
    const std::vector<EdgeTuple>& query_edges,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges,
    std::size_t max_node_sets = 0) {
    py::gil_scoped_release release;
    Graph query = build_graph(query_node_labels, query_edges);
    Graph target = build_graph(target_node_labels, target_edges);
    std::set<std::vector<int>> node_sets;

    auto vertex_comp = [&query, &target](auto query_v, auto target_v) {
        return query[query_v].label == target[target_v].label;
    };
    auto edge_comp = [&query, &target](auto query_e, auto target_e) {
        return query[query_e].label == target[target_e].label;
    };

    boost::vf2_subgraph_mono(
        query,
        target,
        CollectNodeSetsCallback{query, node_sets, max_node_sets},
        boost::vertex_order_by_mult(query),
        boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
    );

    return std::vector<std::vector<int>>(node_sets.begin(), node_sets.end());
}

static std::vector<std::vector<std::vector<int>>> node_sets_subgraph_mono_batch(
    const std::vector<std::vector<int>>& query_node_labels_list,
    const std::vector<std::vector<EdgeTuple>>& query_edges_list,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges,
    std::size_t max_node_sets = 0) {
    py::gil_scoped_release release;
    if (query_node_labels_list.size() != query_edges_list.size()) {
        throw std::runtime_error("query node-label and edge lists must have equal length");
    }

    Graph target = build_graph(target_node_labels, target_edges);
    std::vector<std::vector<std::vector<int>>> all_node_sets;
    all_node_sets.reserve(query_node_labels_list.size());
    for (std::size_t query_index = 0; query_index < query_node_labels_list.size(); ++query_index) {
        Graph query = build_graph(query_node_labels_list[query_index], query_edges_list[query_index]);
        std::set<std::vector<int>> node_sets;

        auto vertex_comp = [&query, &target](auto query_v, auto target_v) {
            return query[query_v].label == target[target_v].label;
        };
        auto edge_comp = [&query, &target](auto query_e, auto target_e) {
            return query[query_e].label == target[target_e].label;
        };

        boost::vf2_subgraph_mono(
            query,
            target,
            CollectNodeSetsCallback{query, node_sets, max_node_sets},
            boost::vertex_order_by_mult(query),
            boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
        );
        all_node_sets.emplace_back(node_sets.begin(), node_sets.end());
    }
    return all_node_sets;
}

static std::vector<std::vector<std::vector<int>>> node_sets_subgraph_iso_batch(
    const std::vector<std::vector<int>>& query_node_labels_list,
    const std::vector<std::vector<EdgeTuple>>& query_edges_list,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges,
    std::size_t max_node_sets = 0) {
    py::gil_scoped_release release;
    if (query_node_labels_list.size() != query_edges_list.size()) {
        throw std::runtime_error("query node-label and edge lists must have equal length");
    }

    Graph target = build_graph(target_node_labels, target_edges);
    std::vector<std::vector<std::vector<int>>> all_node_sets;
    all_node_sets.reserve(query_node_labels_list.size());
    for (std::size_t query_index = 0; query_index < query_node_labels_list.size(); ++query_index) {
        Graph query = build_graph(query_node_labels_list[query_index], query_edges_list[query_index]);
        std::set<std::vector<int>> node_sets;

        auto vertex_comp = [&query, &target](auto query_v, auto target_v) {
            return query[query_v].label == target[target_v].label;
        };
        auto edge_comp = [&query, &target](auto query_e, auto target_e) {
            return query[query_e].label == target[target_e].label;
        };

        boost::vf2_subgraph_iso(
            query,
            target,
            CollectNodeSetsCallback{query, node_sets, max_node_sets},
            boost::vertex_order_by_mult(query),
            boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
        );
        all_node_sets.emplace_back(node_sets.begin(), node_sets.end());
    }
    return all_node_sets;
}

static std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<bool>>
node_sets_subgraph_mono_batch_timed(
    const std::vector<std::vector<int>>& query_node_labels_list,
    const std::vector<std::vector<EdgeTuple>>& query_edges_list,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges,
    std::size_t max_node_sets = 0,
    std::int64_t timeout_ms = 0) {
    py::gil_scoped_release release;
    if (query_node_labels_list.size() != query_edges_list.size()) {
        throw std::runtime_error("query node-label and edge lists must have equal length");
    }

    Graph target = build_graph(target_node_labels, target_edges);
    std::vector<std::vector<std::vector<int>>> all_node_sets;
    std::vector<bool> timed_out(query_node_labels_list.size(), false);
    all_node_sets.reserve(query_node_labels_list.size());

    for (std::size_t query_index = 0; query_index < query_node_labels_list.size(); ++query_index) {
        Graph query = build_graph(query_node_labels_list[query_index], query_edges_list[query_index]);
        std::set<std::vector<int>> node_sets;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

        auto vertex_comp = [&query, &target, timeout_ms, &deadline](auto query_v, auto target_v) {
            if (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                throw Vf2Timeout{};
            }
            return query[query_v].label == target[target_v].label;
        };
        auto edge_comp = [&query, &target, timeout_ms, &deadline](auto query_e, auto target_e) {
            if (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                throw Vf2Timeout{};
            }
            return query[query_e].label == target[target_e].label;
        };

        try {
            boost::vf2_subgraph_mono(
                query,
                target,
                CollectNodeSetsCallback{query, node_sets, max_node_sets},
                boost::vertex_order_by_mult(query),
                boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
            );
        } catch (const Vf2Timeout&) {
            timed_out[query_index] = true;
            node_sets.clear();
        }
        all_node_sets.emplace_back(node_sets.begin(), node_sets.end());
    }
    return {std::move(all_node_sets), std::move(timed_out)};
}

static std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<bool>>
node_sets_subgraph_iso_batch_timed(
    const std::vector<std::vector<int>>& query_node_labels_list,
    const std::vector<std::vector<EdgeTuple>>& query_edges_list,
    const std::vector<int>& target_node_labels,
    const std::vector<EdgeTuple>& target_edges,
    std::size_t max_node_sets = 0,
    std::int64_t timeout_ms = 0) {
    py::gil_scoped_release release;
    if (query_node_labels_list.size() != query_edges_list.size()) {
        throw std::runtime_error("query node-label and edge lists must have equal length");
    }

    Graph target = build_graph(target_node_labels, target_edges);
    std::vector<std::vector<std::vector<int>>> all_node_sets;
    std::vector<bool> timed_out(query_node_labels_list.size(), false);
    all_node_sets.reserve(query_node_labels_list.size());

    for (std::size_t query_index = 0; query_index < query_node_labels_list.size(); ++query_index) {
        Graph query = build_graph(query_node_labels_list[query_index], query_edges_list[query_index]);
        std::set<std::vector<int>> node_sets;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

        auto vertex_comp = [&query, &target, timeout_ms, &deadline](auto query_v, auto target_v) {
            if (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                throw Vf2Timeout{};
            }
            return query[query_v].label == target[target_v].label;
        };
        auto edge_comp = [&query, &target, timeout_ms, &deadline](auto query_e, auto target_e) {
            if (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                throw Vf2Timeout{};
            }
            return query[query_e].label == target[target_e].label;
        };

        try {
            boost::vf2_subgraph_iso(
                query,
                target,
                CollectNodeSetsCallback{query, node_sets, max_node_sets},
                boost::vertex_order_by_mult(query),
                boost::vertices_equivalent(vertex_comp).edges_equivalent(edge_comp)
            );
        } catch (const Vf2Timeout&) {
            timed_out[query_index] = true;
            node_sets.clear();
        }
        all_node_sets.emplace_back(node_sets.begin(), node_sets.end());
    }
    return {std::move(all_node_sets), std::move(timed_out)};
}

PYBIND11_MODULE(boost_vf2, m) {
    m.doc() = "Boost.Graph VF2 subgraph monomorphism matcher with node and edge labels";
    m.def(
        "contains_subgraph",
        &contains_subgraph,
        py::arg("query_node_labels"),
        py::arg("query_edges"),
        py::arg("target_node_labels"),
        py::arg("target_edges")
    );
    m.def(
        "node_sets_subgraph_mono",
        &node_sets_subgraph_mono,
        py::arg("query_node_labels"),
        py::arg("query_edges"),
        py::arg("target_node_labels"),
        py::arg("target_edges"),
        py::arg("max_node_sets") = 0
    );
    m.def(
        "node_sets_subgraph_mono_batch",
        &node_sets_subgraph_mono_batch,
        py::arg("query_node_labels_list"),
        py::arg("query_edges_list"),
        py::arg("target_node_labels"),
        py::arg("target_edges"),
        py::arg("max_node_sets") = 0
    );
    m.def(
        "node_sets_subgraph_iso_batch",
        &node_sets_subgraph_iso_batch,
        py::arg("query_node_labels_list"),
        py::arg("query_edges_list"),
        py::arg("target_node_labels"),
        py::arg("target_edges"),
        py::arg("max_node_sets") = 0
    );
    m.def(
        "node_sets_subgraph_mono_batch_timed",
        &node_sets_subgraph_mono_batch_timed,
        py::arg("query_node_labels_list"),
        py::arg("query_edges_list"),
        py::arg("target_node_labels"),
        py::arg("target_edges"),
        py::arg("max_node_sets") = 0,
        py::arg("timeout_ms") = 0
    );
    m.def(
        "node_sets_subgraph_iso_batch_timed",
        &node_sets_subgraph_iso_batch_timed,
        py::arg("query_node_labels_list"),
        py::arg("query_edges_list"),
        py::arg("target_node_labels"),
        py::arg("target_edges"),
        py::arg("max_node_sets") = 0,
        py::arg("timeout_ms") = 0
    );
}
