// kwsbin/kws-search.cc

// Copyright 2012-2015  Johns Hopkins University (Authors: Guoguo Chen,
//                                                         Daniel Povey.
//                                                         Yenda Trmal)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "kws/kaldi-kws.h"
#include "kws/kws-functions.h"

void WriteKWSResults(
    const std::string &kwid,
    const std::vector<std::tuple<kaldi::int32, kaldi::int32, kaldi::int32, double> > &results,
    const std::vector<std::vector<kaldi::int32> > *paths,
    int32 frame_subsampling_factor,
    double negative_tolerance,
    kaldi::TableWriter<kaldi::BasicVectorHolder<double> > *writer) {
  if (paths != nullptr) {
    KALDI_ASSERT(results.size() == (*paths).size());
  }
  kaldi::int32 uid, tbeg, tend;
  double score;
  for (int i = 0; i < results.size(); ++i) {
    std::tie(uid, tbeg, tend, score) = results[i];
    std::vector<double> output;
    output.push_back(uid);
    output.push_back(tbeg * frame_subsampling_factor);
    output.push_back(tend * frame_subsampling_factor);
    if (score < 0) {
      if (score < negative_tolerance) {
        KALDI_WARN << "Score out of expected range: " << score;
      }
      score = 0.0;
    }
    output.push_back(score);
    if (paths != nullptr) {
      for (int j = 0; j < (*paths)[i].size(); ++j)
        output.push_back((*paths)[i][j]);
    }
    writer->Write(kwid, output);
  }
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using std::vector;
    typedef kaldi::int32 int32;
    typedef kaldi::uint32 uint32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Search the keywords over the index. This program can be executed\n"
        "in parallel, either on the index side or the keywords side; we use\n"
        "a script to combine the final search results. Note that the index\n"
        "archive has a single key \"global\".\n\n"
        "Search has one or two outputs. The first one is mandatory and will\n"
        "contain the seach output, i.e. list of all found keyword instances\n"
        "The file is in the following format:\n"
        "kw_id utt_id beg_frame end_frame neg_logprob\n"
        " e.g.: \n"
        "KW105-0198 7 335 376 1.91254\n\n"
        "The second parameter is optional and allows the user to gather more\n"
        "statistics about the individual instances from the posting list.\n"
        "Remember \"keyword\" is an FST and as such, there can be multiple\n"
        "paths matching in the keyword and in the lattice index in that given\n"
        "time period. The stats output will provide all matching paths\n"
        "each with the appropriate score. \n"
        "The format is as follows:\n"
        "kw_id utt_id beg_frame end_frame neg_logprob 0 w_id1 w_id2 ... 0\n"
        " e.g.: \n"
        "KW105-0198 7 335 376 16.01254 0 5766 5659 0\n"
        "\n"
        "Usage: kws-search [options] <index-rspecifier> <keywords-rspecifier> "
        "<results-wspecifier> [<stats_wspecifier>]\n"
        " e.g.: kws-search ark:index.idx ark:keywords.fsts "
                           "ark:results ark:stats\n";

    ParseOptions po(usage);

    int32 n_best = -1;
    int32 keyword_nbest = -1;
    bool strict = true;
    double negative_tolerance = -0.1;
    double keyword_beam = -1;
    int32 frame_subsampling_factor = 1;

    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "Frame subsampling factor. (Default value 1)");
    po.Register("nbest", &n_best, "Return the best n hypotheses.");
    po.Register("keyword-nbest", &keyword_nbest,
                "Pick the best n keywords if the FST contains "
                "multiple keywords.");
    po.Register("strict", &strict, "Affects the return status of the program.");
    po.Register("negative-tolerance", &negative_tolerance,
                "The program will print a warning if we get negative score "
                "smaller than this tolerance.");
    po.Register("keyword-beam", &keyword_beam,
                "Prune the FST with the given beam if the FST contains "
                "multiple keywords.");

    if (n_best < 0 && n_best != -1) {
      KALDI_ERR << "Bad number for nbest";
      exit(1);
    }
    if (keyword_nbest < 0 && keyword_nbest != -1) {
      KALDI_ERR << "Bad number for keyword-nbest";
      exit(1);
    }
    if (keyword_beam < 0 && keyword_beam != -1) {
      KALDI_ERR << "Bad number for keyword-beam";
      exit(1);
    }

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string index_rspecifier = po.GetArg(1),
        keyword_rspecifier = po.GetArg(2),
        results_wspecifier = po.GetArg(3),
        stats_wspecifier = po.GetOptArg(4);

    RandomAccessTableReader< VectorFstTplHolder<KwsLexicographicArc> >
                                                index_reader(index_rspecifier);
    SequentialTableReader<VectorFstHolder> keyword_reader(keyword_rspecifier);
    TableWriter<BasicVectorHolder<double> > results_writer(results_wspecifier),
                                            stats_writer(stats_wspecifier);

    // Index has key "global"
    KwsLexicographicFst index = index_reader.Value("global");

    // First we have to remove the disambiguation symbols. But rather than
    // removing them totally, we actually move them from input side to output
    // side, making the output symbol a "combined" symbol of the disambiguation
    // symbols and the utterance id's.
    // Note that in Dogan and Murat's original paper, they simply remove the
    // disambiguation symbol on the input symbol side, which will not allow us
    // to do epsilon removal after composition with the keyword FST. They have
    // to traverse the resulting FST.
    fst::internal::EncodeTable<KwsLexicographicArc> encode_table(kEncodeLabels);
    EncodeKwsDisambiguationSymbols(&index, &encode_table);

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      VectorFst<StdArc> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      // Process the case where we have confusion for keywords
      if (keyword_beam != -1) {
        Prune(&keyword, keyword_beam);
      }
      if (keyword_nbest != -1) {
        VectorFst<StdArc> tmp;
        ShortestPath(keyword, &tmp, keyword_nbest, true, true);
        keyword = tmp;
      }

      bool success = true;
      std::vector<std::tuple<int32, int32, int32, double> > results;

      if (stats_wspecifier != "") {
        KwsLexicographicFst matched_seq;
        success = SearchKwsIndex(index, keyword, encode_table, n_best,
                                 &results, &matched_seq);
        std::vector<std::tuple<int32, int32, int32, double> > stats;
        std::vector<std::vector<int32> > paths;
        ComputeDetailedStatistics(matched_seq, encode_table, &stats, &paths);
        WriteKWSResults(key, stats, &paths, frame_subsampling_factor,
                        negative_tolerance, &stats_writer);
      } else {
        success = SearchKwsIndex(index, keyword, encode_table, n_best,
                                 &results);
      }

      if (!success) {
        KALDI_WARN << "Search failed for key " << key;
        n_fail++;
        continue;
      }

      WriteKWSResults(key, results, nullptr, frame_subsampling_factor,
                      negative_tolerance, &results_writer);
      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " keywords";
    if (strict == true)
      return (n_done != 0 ? 0 : 1);
    else
      return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
