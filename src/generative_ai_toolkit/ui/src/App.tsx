/**
 * Copyright 2024 Amazon.com, Inc. and its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React from 'react';
import { Routes, Route } from 'react-router-dom';
import TestPage from './pages/TestPage';
import HomePage from './pages/HomePage';
import './App.css';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/ui" element={<HomePage />} />
      <Route path="/ui/test" element={<TestPage />} />
    </Routes>
  );
};

export default App;
