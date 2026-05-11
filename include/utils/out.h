#pragma once

struct Out {
  long long pop_time = 0;
  long long search_time = 0;

  long long twohop_time = 0;
  long long filter_time = 0;
  long long comp_time = 0;
  long long bk_time = 0;

  long long checked_count = 0;
  int twohop_count = 0;
};