/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Author: Zhe Sun, Ahmet Erdem
// April 20, 2017

#ifndef UTILS_CPPCLASS_H
#define UTILS_CPPCLASS_H

extern void sparse_dot_topn_parallel(int n_row,
      	              int n_col,
      	              int Ap[],
      	              int Aj[],
      	              double Ax[],
      	              int Bp[],
      	              int Bj[],
      	              double Bx[],
                      int ntop,
                      double lower_bound,
      	                    int Cp[],
      	                    int Cj[],
      	                    double Cx[],
      	                    int n_jobs);

#endif //UTILS_CPPCLASS_H
